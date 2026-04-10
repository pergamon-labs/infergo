package httpserver_test

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/pergamon-labs/infergo/infer/httpserver"
	"github.com/pergamon-labs/infergo/infer/packs"
)

func BenchmarkMetadataTextPackInfergoBasicSST2(b *testing.B) {
	pack, err := packs.LoadTextPack("infergo-basic-sst2")
	if err != nil {
		b.Fatalf("LoadTextPack() error = %v", err)
	}
	defer pack.Close()

	handler := httpserver.NewTextPackMux(pack)

	b.ReportAllocs()
	b.ResetTimer()

	for b.Loop() {
		req := httptest.NewRequest(http.MethodGet, "/metadata", nil)
		rec := httptest.NewRecorder()
		handler.ServeHTTP(rec, req)
		if rec.Code != http.StatusOK {
			b.Fatalf("metadata status = %d, want %d", rec.Code, http.StatusOK)
		}
	}
}

func BenchmarkPredictTextInfergoBasicSST2HTTP(b *testing.B) {
	pack, err := packs.LoadTextPack("infergo-basic-sst2")
	if err != nil {
		b.Fatalf("LoadTextPack() error = %v", err)
	}
	defer pack.Close()

	handler := httpserver.NewTextPackMux(pack)
	body := []byte(`{"text":"This product is excellent and reliable."}`)

	b.ReportAllocs()
	b.ResetTimer()

	for b.Loop() {
		req := httptest.NewRequest(http.MethodPost, "/predict", bytes.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		rec := httptest.NewRecorder()
		handler.ServeHTTP(rec, req)
		if rec.Code != http.StatusOK {
			b.Fatalf("predict status = %d, want %d", rec.Code, http.StatusOK)
		}
	}
}

func BenchmarkPredictTokenTextInfergoBasicFrenchNERHTTP(b *testing.B) {
	pack, err := packs.LoadTokenPack("infergo-basic-french-ner")
	if err != nil {
		b.Fatalf("LoadTokenPack() error = %v", err)
	}
	defer pack.Close()

	handler := httpserver.NewTokenPackMux(pack)
	body := []byte(`{"text":"Sophie Tremblay a parlé avec Hydro-Québec à Montréal."}`)

	b.ReportAllocs()
	b.ResetTimer()

	for b.Loop() {
		req := httptest.NewRequest(http.MethodPost, "/predict", bytes.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		rec := httptest.NewRecorder()
		handler.ServeHTTP(rec, req)
		if rec.Code != http.StatusOK {
			b.Fatalf("predict status = %d, want %d", rec.Code, http.StatusOK)
		}
	}
}

func BenchmarkPredictTokenTokensInfergoBasicFrenchNERHTTP(b *testing.B) {
	pack, err := packs.LoadTokenPack("infergo-basic-french-ner")
	if err != nil {
		b.Fatalf("LoadTokenPack() error = %v", err)
	}
	defer pack.Close()

	handler := httpserver.NewTokenPackMux(pack)
	body := []byte(`{"tokens":["sophie","tremblay","a","parlé","avec","hydro","québec","à","montréal"]}`)

	b.ReportAllocs()
	b.ResetTimer()

	for b.Loop() {
		req := httptest.NewRequest(http.MethodPost, "/predict", bytes.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		rec := httptest.NewRecorder()
		handler.ServeHTTP(rec, req)
		if rec.Code != http.StatusOK {
			b.Fatalf("predict status = %d, want %d", rec.Code, http.StatusOK)
		}
	}
}
