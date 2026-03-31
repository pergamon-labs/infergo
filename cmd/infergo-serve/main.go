package main

import (
	"flag"
	"log"
	"net/http"
	"strings"

	"github.com/pergamon-labs/infergo/infer/httpserver"
	"github.com/pergamon-labs/infergo/infer/packs"
)

func main() {
	addr := flag.String("addr", ":8080", "http listen address")
	task := flag.String("task", "text", "which serving task to expose: text or token")
	packKey := flag.String("pack", "", "supported checked-in pack key")
	flag.Parse()

	switch *task {
	case "text":
		key := *packKey
		if key == "" {
			key = "infergo-basic-sst2"
		}
		pack, err := packs.LoadTextPack(key)
		if err != nil {
			log.Fatalf("load text pack: %v", err)
		}
		defer pack.Close()

		mux := httpserver.NewTextPackMux(pack)
		logServeHints(*addr, "text", key)
		if err := http.ListenAndServe(*addr, mux); err != nil {
			log.Fatal(err)
		}
	case "token":
		key := *packKey
		if key == "" {
			key = "infergo-basic-french-ner"
		}
		pack, err := packs.LoadTokenPack(key)
		if err != nil {
			log.Fatalf("load token pack: %v", err)
		}
		defer pack.Close()

		mux := httpserver.NewTokenPackMux(pack)
		logServeHints(*addr, "token", key)
		if err := http.ListenAndServe(*addr, mux); err != nil {
			log.Fatal(err)
		}
	default:
		log.Fatalf("unsupported task %q; expected text or token", *task)
	}
}

func logServeHints(addr, task, pack string) {
	curlAddr := addr
	if strings.HasPrefix(curlAddr, ":") {
		curlAddr = "127.0.0.1" + curlAddr
	}
	log.Printf("InferGo serving %s pack %q on %s", task, pack, addr)
	log.Printf("Health: curl -s http://%s/healthz | jq", curlAddr)
	log.Printf("Metadata: curl -s http://%s/metadata | jq", curlAddr)
	switch task {
	case "text":
		log.Printf("Predict: curl -s -X POST http://%s/predict -H 'Content-Type: application/json' -d '{\"text\":\"This product is excellent and reliable.\"}' | jq", curlAddr)
	case "token":
		log.Printf("Predict: curl -s -X POST http://%s/predict -H 'Content-Type: application/json' -d '{\"text\":\"Sophie Tremblay a parlé avec Hydro-Québec à Montréal.\"}' | jq", curlAddr)
	}
}
