package main

import (
	"context"
	"errors"
	"flag"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/pergamon-labs/infergo/backends/bionet"
	"github.com/pergamon-labs/infergo/infer"
	"github.com/pergamon-labs/infergo/infer/httpserver"
	"github.com/pergamon-labs/infergo/infer/packs"
)

func main() {
	serverCfg := httpserver.DefaultServerConfig()

	addr := flag.String("addr", envString("INFERGO_SERVE_ADDR", serverCfg.Addr), "http listen address")
	task := flag.String("task", envString("INFERGO_SERVE_TASK", "text"), "which serving task to expose: text or token")
	packKey := flag.String("pack", envString("INFERGO_SERVE_PACK", ""), "supported checked-in pack key")
	bundleDir := flag.String("bundle", envString("INFERGO_SERVE_BUNDLE", ""), "bundle directory to serve directly without using curated pack manifests")
	logRequests := flag.Bool("log-requests", envBool("INFERGO_SERVE_LOG_REQUESTS", true), "log one line per request")
	readTimeout := flag.Duration("read-timeout", envDuration("INFERGO_SERVE_READ_TIMEOUT", serverCfg.ReadTimeout), "http read timeout")
	readHeaderTimeout := flag.Duration("read-header-timeout", envDuration("INFERGO_SERVE_READ_HEADER_TIMEOUT", serverCfg.ReadHeaderTimeout), "http read-header timeout")
	writeTimeout := flag.Duration("write-timeout", envDuration("INFERGO_SERVE_WRITE_TIMEOUT", serverCfg.WriteTimeout), "http write timeout")
	idleTimeout := flag.Duration("idle-timeout", envDuration("INFERGO_SERVE_IDLE_TIMEOUT", serverCfg.IdleTimeout), "http idle timeout")
	shutdownTimeout := flag.Duration("shutdown-timeout", envDuration("INFERGO_SERVE_SHUTDOWN_TIMEOUT", serverCfg.ShutdownTimeout), "graceful shutdown timeout")
	flag.Parse()

	serverCfg.Addr = *addr
	serverCfg.ReadTimeout = *readTimeout
	serverCfg.ReadHeaderTimeout = *readHeaderTimeout
	serverCfg.WriteTimeout = *writeTimeout
	serverCfg.IdleTimeout = *idleTimeout
	serverCfg.ShutdownTimeout = *shutdownTimeout

	options := []httpserver.Option{
		httpserver.WithLogger(log.Default()),
		httpserver.WithRequestLogging(*logRequests),
	}

	if *packKey != "" && *bundleDir != "" {
		log.Fatal("use only one of -pack or -bundle")
	}

	switch *task {
	case "text":
		if *bundleDir != "" {
			classifier, err := infer.LoadTextClassifier(*bundleDir)
			if err != nil {
				log.Fatalf("load text bundle: %v", err)
			}
			defer classifier.Close()

			info, err := bionet.InspectTextClassificationBundle(*bundleDir)
			if err != nil {
				log.Fatalf("inspect text bundle: %v", err)
			}

			mux := httpserver.NewTextClassifierMux(classifier, httpserver.TextClassifierMetadata{
				ModelID:                info.ModelID,
				SupportsRawText:        info.SupportsRawText,
				SupportsPairText:       info.SupportsPairText,
				SupportsTokenizedInput: info.SupportsTokenizedInput,
			}, options...)
			serve(serverCfg, mux, "text-bundle", *bundleDir)
			return
		}

		key := *packKey
		if key == "" {
			key = "infergo-basic-sst2"
		}
		pack, err := packs.LoadTextPack(key)
		if err != nil {
			log.Fatalf("load text pack: %v", err)
		}
		defer pack.Close()

		mux := httpserver.NewTextPackMux(pack, options...)
		serve(serverCfg, mux, "text", key)
	case "token":
		if *bundleDir != "" {
			log.Fatal("bundle serving is currently implemented only for text-classification bundles")
		}

		key := *packKey
		if key == "" {
			key = "infergo-basic-french-ner"
		}
		pack, err := packs.LoadTokenPack(key)
		if err != nil {
			log.Fatalf("load token pack: %v", err)
		}
		defer pack.Close()

		mux := httpserver.NewTokenPackMux(pack, options...)
		serve(serverCfg, mux, "token", key)
	default:
		log.Fatalf("unsupported task %q; expected text or token", *task)
	}
}

func serve(cfg httpserver.ServerConfig, handler http.Handler, task, pack string) {
	server := httpserver.NewServer(handler, cfg)
	logServeHints(cfg, task, pack)

	errCh := make(chan error, 1)
	go func() {
		errCh <- server.ListenAndServe()
	}()

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	select {
	case err := <-errCh:
		if err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Fatal(err)
		}
	case <-ctx.Done():
		log.Printf("InferGo shutting down after signal: %v", ctx.Err())
		shutdownCtx, cancel := context.WithTimeout(context.Background(), cfg.ShutdownTimeout)
		defer cancel()
		if err := server.Shutdown(shutdownCtx); err != nil {
			log.Fatalf("graceful shutdown failed: %v", err)
		}
		if err := <-errCh; err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Fatal(err)
		}
	}
}

func logServeHints(cfg httpserver.ServerConfig, task, pack string) {
	addr := cfg.Addr
	curlAddr := addr
	if strings.HasPrefix(curlAddr, ":") {
		curlAddr = "127.0.0.1" + curlAddr
	}
	log.Printf("InferGo serving %s pack %q on %s", task, pack, addr)
	log.Printf("Timeouts: read=%s read-header=%s write=%s idle=%s shutdown=%s", cfg.ReadTimeout, cfg.ReadHeaderTimeout, cfg.WriteTimeout, cfg.IdleTimeout, cfg.ShutdownTimeout)
	log.Printf("Health: curl -s http://%s/healthz | jq", curlAddr)
	log.Printf("Metadata: curl -s http://%s/metadata | jq", curlAddr)
	switch task {
	case "text":
		log.Printf("Predict: curl -s -X POST http://%s/predict -H 'Content-Type: application/json' -d '{\"text\":\"This product is excellent and reliable.\"}' | jq", curlAddr)
	case "text-bundle":
		log.Printf("Predict: curl -s -X POST http://%s/predict -H 'Content-Type: application/json' -d '{\"input_ids\":[101,2023,2003,1037,2742,102],\"attention_mask\":[1,1,1,1,1,1]}' | jq", curlAddr)
	case "token":
		log.Printf("Predict: curl -s -X POST http://%s/predict -H 'Content-Type: application/json' -d '{\"text\":\"Sophie Tremblay a parlé avec Hydro-Québec à Montréal.\"}' | jq", curlAddr)
	}
}

func envString(key, fallback string) string {
	if value := strings.TrimSpace(os.Getenv(key)); value != "" {
		return value
	}
	return fallback
}

func envBool(key string, fallback bool) bool {
	value := strings.TrimSpace(os.Getenv(key))
	if value == "" {
		return fallback
	}
	parsed, err := strconv.ParseBool(value)
	if err != nil {
		log.Fatalf("parse %s as bool: %v", key, err)
	}
	return parsed
}

func envDuration(key string, fallback time.Duration) time.Duration {
	value := strings.TrimSpace(os.Getenv(key))
	if value == "" {
		return fallback
	}
	parsed, err := time.ParseDuration(value)
	if err != nil {
		log.Fatalf("parse %s as duration: %v", key, err)
	}
	return parsed
}
