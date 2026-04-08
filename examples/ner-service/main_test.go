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

	entities := groupEntities(tokens, labels)
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
	}, payload.Entities)
}
