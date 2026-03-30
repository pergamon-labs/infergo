package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"text/tabwriter"

	curatedpacks "github.com/pergamon-labs/infergo/infer/packs"
)

type output struct {
	Text  []curatedTextPack  `json:"text_packs,omitempty"`
	Token []curatedTokenPack `json:"token_packs,omitempty"`
}

type curatedTextPack struct {
	Key              string `json:"key"`
	ModelID          string `json:"model_id"`
	DefaultBundleKey string `json:"default_bundle_key"`
	DefaultBundleDir string `json:"default_bundle_dir"`
	ReferencePath    string `json:"reference_path"`
	SupportsRawText  bool   `json:"supports_raw_text"`
}

type curatedTokenPack struct {
	Key             string `json:"key"`
	ModelID         string `json:"model_id"`
	NativeBundleDir string `json:"native_bundle_dir"`
	ReferencePath   string `json:"reference_path"`
}

func main() {
	task := flag.String("task", "all", "which pack family to print: all, text, or token")
	jsonOutput := flag.Bool("json", false, "print machine-readable JSON instead of a table")
	rawTextOnly := flag.Bool("raw-text-only", false, "only print text packs that support raw text")
	flag.Parse()

	if *task != "all" && *task != "text" && *task != "token" {
		log.Fatalf("unsupported task %q; expected all, text, or token", *task)
	}

	cwd, err := os.Getwd()
	if err != nil {
		log.Fatalf("resolve current working directory: %v", err)
	}

	textPacks, err := curatedpacks.ListTextPacks()
	if err != nil {
		log.Fatalf("list text packs: %v", err)
	}

	tokenPacks, err := curatedpacks.ListTokenPacks()
	if err != nil {
		log.Fatalf("list token packs: %v", err)
	}

	if *rawTextOnly {
		filtered := make([]curatedpacks.TextPackInfo, 0, len(textPacks))
		for _, item := range textPacks {
			if item.SupportsRawText {
				filtered = append(filtered, item)
			}
		}
		textPacks = filtered
	}

	if *jsonOutput {
		if err := writeJSON(os.Stdout, cwd, *task, textPacks, tokenPacks); err != nil {
			log.Fatalf("write json: %v", err)
		}
		return
	}

	if err := writeTable(os.Stdout, cwd, *task, textPacks, tokenPacks); err != nil {
		log.Fatalf("write table: %v", err)
	}
}

func writeJSON(dst io.Writer, cwd, task string, textPacks []curatedpacks.TextPackInfo, tokenPacks []curatedpacks.TokenPackInfo) error {
	payload := output{}
	if task == "all" || task == "text" {
		payload.Text = make([]curatedTextPack, 0, len(textPacks))
		for _, item := range textPacks {
			payload.Text = append(payload.Text, curatedTextPack{
				Key:              item.Key,
				ModelID:          item.ModelID,
				DefaultBundleKey: item.DefaultBundleKey,
				DefaultBundleDir: displayPath(cwd, item.DefaultBundleDir),
				ReferencePath:    displayPath(cwd, item.ReferencePath),
				SupportsRawText:  item.SupportsRawText,
			})
		}
	}
	if task == "all" || task == "token" {
		payload.Token = make([]curatedTokenPack, 0, len(tokenPacks))
		for _, item := range tokenPacks {
			payload.Token = append(payload.Token, curatedTokenPack{
				Key:             item.Key,
				ModelID:         item.ModelID,
				NativeBundleDir: displayPath(cwd, item.NativeBundleDir),
				ReferencePath:   displayPath(cwd, item.ReferencePath),
			})
		}
	}

	encoder := json.NewEncoder(dst)
	encoder.SetIndent("", "  ")
	return encoder.Encode(payload)
}

func writeTable(dst io.Writer, cwd, task string, textPacks []curatedpacks.TextPackInfo, tokenPacks []curatedpacks.TokenPackInfo) error {
	writer := tabwriter.NewWriter(dst, 0, 8, 2, ' ', 0)
	defer writer.Flush()

	if task == "all" || task == "text" {
		fmt.Fprintln(writer, "TEXT PACKS")
		fmt.Fprintln(writer, "KEY\tMODEL ID\tRAW TEXT\tDEFAULT BUNDLE\tREFERENCE")
		for _, item := range textPacks {
			rawText := "no"
			if item.SupportsRawText {
				rawText = "yes"
			}
			fmt.Fprintf(writer, "%s\t%s\t%s\t%s\t%s\n", item.Key, item.ModelID, rawText, item.DefaultBundleKey, displayPath(cwd, item.ReferencePath))
		}
		if len(textPacks) == 0 {
			fmt.Fprintln(writer, "(none)")
		}
	}

	if task == "all" || task == "token" {
		if task == "all" {
			fmt.Fprintln(writer)
		}
		fmt.Fprintln(writer, "TOKEN PACKS")
		fmt.Fprintln(writer, "KEY\tMODEL ID\tNATIVE BUNDLE\tREFERENCE")
		for _, item := range tokenPacks {
			fmt.Fprintf(writer, "%s\t%s\t%s\t%s\n", item.Key, item.ModelID, displayPath(cwd, item.NativeBundleDir), displayPath(cwd, item.ReferencePath))
		}
		if len(tokenPacks) == 0 {
			fmt.Fprintln(writer, "(none)")
		}
	}

	return nil
}

func displayPath(cwd, value string) string {
	if value == "" {
		return value
	}
	if !filepath.IsAbs(value) {
		return filepath.Clean(filepath.ToSlash(value))
	}
	rel, err := filepath.Rel(cwd, value)
	if err != nil {
		return filepath.Clean(filepath.ToSlash(value))
	}
	return filepath.Clean(filepath.ToSlash(rel))
}
