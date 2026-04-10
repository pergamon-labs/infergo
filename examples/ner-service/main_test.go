package main

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/pergamon-labs/infergo/infer/packs"
	"github.com/stretchr/testify/require"
)

func TestGroupEntities(t *testing.T) {
	tokens := []string{"sophie", "tremblay", "a", "parlé", "avec", "hydro", "-", "québec", "à", "montréal"}
	labels := []string{"I-PER", "I-PER", "O", "O", "O", "I-ORG", "I-ORG", "I-ORG", "O", "I-LOC"}

	entities := groupEntities("", tokens, labels, nil)
	require.Equal(t, []namedEntity{
		{
			Label:      "PER",
			Text:       "sophie tremblay",
			Tokens:     []string{"sophie", "tremblay"},
			StartToken: 0,
			EndToken:   1,
		},
		{
			Label:      "ORG",
			Text:       "hydro-québec",
			Tokens:     []string{"hydro", "-", "québec"},
			StartToken: 5,
			EndToken:   7,
		},
		{
			Label:      "LOC",
			Text:       "montréal",
			Tokens:     []string{"montréal"},
			StartToken: 9,
			EndToken:   9,
		},
	}, entities)
}

func TestNERServiceExtractText(t *testing.T) {
	pack, err := packs.LoadTokenPack("infergo-basic-french-ner")
	require.NoError(t, err)
	t.Cleanup(func() { require.NoError(t, pack.Close()) })

	server := httptest.NewServer((&service{pack: pack}).newMux())
	t.Cleanup(server.Close)

	resp, err := http.Post(server.URL+"/extract", "application/json", bytes.NewBufferString(`{"text":"Sophie Tremblay a parlé avec Hydro-Québec à Montréal."}`))
	require.NoError(t, err)
	defer resp.Body.Close()
	require.Equal(t, http.StatusOK, resp.StatusCode)

	var payload extractResponse
	require.NoError(t, json.NewDecoder(resp.Body).Decode(&payload))
	require.Equal(t, "infergo-basic-french-ner", payload.PackKey)
	require.NotEmpty(t, payload.TokenLabels)
	require.Equal(t, []namedEntity{
		{
			Label:      "PER",
			Text:       "Sophie Tremblay",
			Tokens:     []string{"sophie", "tremblay"},
			StartToken: 0,
			EndToken:   1,
			Span: &textSpan{
				StartByte: 0,
				EndByte:   15,
				StartChar: 0,
				EndChar:   15,
			},
		},
		{
			Label:      "ORG",
			Text:       "Hydro-Québec",
			Tokens:     []string{"hydro", "-", "québec"},
			StartToken: 5,
			EndToken:   7,
			Span: &textSpan{
				StartByte: 30,
				EndByte:   43,
				StartChar: 29,
				EndChar:   41,
			},
		},
		{
			Label:      "LOC",
			Text:       "Montréal",
			Tokens:     []string{"montréal"},
			StartToken: 9,
			EndToken:   9,
			Span: &textSpan{
				StartByte: 47,
				EndByte:   56,
				StartChar: 44,
				EndChar:   52,
			},
		},
	}, payload.Entities)
	require.Equal(t, []packs.TokenSpan{
		{Token: "sophie", StartByte: 0, EndByte: 6, StartChar: 0, EndChar: 6},
		{Token: "tremblay", StartByte: 7, EndByte: 15, StartChar: 7, EndChar: 15},
		{Token: "a", StartByte: 16, EndByte: 17, StartChar: 16, EndChar: 17},
		{Token: "parlé", StartByte: 18, EndByte: 24, StartChar: 18, EndChar: 23},
		{Token: "avec", StartByte: 25, EndByte: 29, StartChar: 24, EndChar: 28},
		{Token: "hydro", StartByte: 30, EndByte: 35, StartChar: 29, EndChar: 34},
		{Token: "-", StartByte: 35, EndByte: 36, StartChar: 34, EndChar: 35},
		{Token: "québec", StartByte: 36, EndByte: 43, StartChar: 35, EndChar: 41},
		{Token: "à", StartByte: 44, EndByte: 46, StartChar: 42, EndChar: 43},
		{Token: "montréal", StartByte: 47, EndByte: 56, StartChar: 44, EndChar: 52},
	}, payload.TokenSpans)
}
