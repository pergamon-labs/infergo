package packs_test

import (
	"testing"

	"github.com/pergamon-labs/infergo/infer/packs"
)

func BenchmarkLoadTextPackInfergoBasicSST2(b *testing.B) {
	b.ReportAllocs()

	for b.Loop() {
		pack, err := packs.LoadTextPack("infergo-basic-sst2")
		if err != nil {
			b.Fatalf("LoadTextPack() error = %v", err)
		}
		if err := pack.Close(); err != nil {
			b.Fatalf("Close() error = %v", err)
		}
	}
}

func BenchmarkPredictTextInfergoBasicSST2(b *testing.B) {
	pack, err := packs.LoadTextPack("infergo-basic-sst2")
	if err != nil {
		b.Fatalf("LoadTextPack() error = %v", err)
	}
	defer pack.Close()

	b.ReportAllocs()
	b.ResetTimer()

	for b.Loop() {
		if _, err := pack.PredictText("This product is excellent and reliable."); err != nil {
			b.Fatalf("PredictText() error = %v", err)
		}
	}
}

func BenchmarkLoadTokenPackInfergoBasicFrenchNER(b *testing.B) {
	b.ReportAllocs()

	for b.Loop() {
		pack, err := packs.LoadTokenPack("infergo-basic-french-ner")
		if err != nil {
			b.Fatalf("LoadTokenPack() error = %v", err)
		}
		if err := pack.Close(); err != nil {
			b.Fatalf("Close() error = %v", err)
		}
	}
}

func BenchmarkPredictTextInfergoBasicFrenchNER(b *testing.B) {
	pack, err := packs.LoadTokenPack("infergo-basic-french-ner")
	if err != nil {
		b.Fatalf("LoadTokenPack() error = %v", err)
	}
	defer pack.Close()

	b.ReportAllocs()
	b.ResetTimer()

	for b.Loop() {
		if _, err := pack.PredictText("Sophie Tremblay a parlé avec Hydro-Québec à Montréal."); err != nil {
			b.Fatalf("PredictText() error = %v", err)
		}
	}
}

func BenchmarkPredictTokensInfergoBasicFrenchNER(b *testing.B) {
	pack, err := packs.LoadTokenPack("infergo-basic-french-ner")
	if err != nil {
		b.Fatalf("LoadTokenPack() error = %v", err)
	}
	defer pack.Close()

	tokens := []string{"sophie", "tremblay", "a", "parlé", "avec", "hydro", "québec", "à", "montréal"}

	b.ReportAllocs()
	b.ResetTimer()

	for b.Loop() {
		if _, err := pack.PredictTokens(tokens); err != nil {
			b.Fatalf("PredictTokens() error = %v", err)
		}
	}
}
