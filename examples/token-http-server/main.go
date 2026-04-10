package main

import (
	"flag"
	"log"
	"net/http"

	"github.com/pergamon-labs/infergo/infer/httpserver"
	"github.com/pergamon-labs/infergo/infer/packs"
)

func main() {
	addr := flag.String("addr", ":8081", "http listen address")
	packKey := flag.String("pack", "infergo-basic-french-ner", "supported checked-in token pack key")
	flag.Parse()

	pack, err := packs.LoadTokenPack(*packKey)
	if err != nil {
		log.Fatalf("load token pack: %v", err)
	}
	defer pack.Close()

	log.Printf("InferGo token example server listening on %s", *addr)
	log.Printf("This example now reuses the stable infer/httpserver package.")
	if err := http.ListenAndServe(*addr, httpserver.NewTokenPackMux(pack)); err != nil {
		log.Fatal(err)
	}
}
