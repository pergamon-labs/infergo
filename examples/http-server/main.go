package main

import (
	"flag"
	"log"
	"net/http"

	"github.com/pergamon-labs/infergo/infer/httpserver"
	"github.com/pergamon-labs/infergo/infer/packs"
)

func main() {
	addr := flag.String("addr", ":8080", "http listen address")
	packKey := flag.String("pack", "infergo-basic-sst2", "supported checked-in text pack key")
	flag.Parse()

	pack, err := packs.LoadTextPack(*packKey)
	if err != nil {
		log.Fatalf("load text pack: %v", err)
	}
	defer pack.Close()

	log.Printf("InferGo example server listening on %s", *addr)
	log.Printf("This example now reuses the stable infer/httpserver package.")
	if err := http.ListenAndServe(*addr, httpserver.NewTextPackMux(pack)); err != nil {
		log.Fatal(err)
	}
}
