package httpserver

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/pergamon-labs/infergo/infer/packs"
	"github.com/stretchr/testify/require"
)

func TestNewTextPackMuxMetadataAndPredict(t *testing.T) {
	pack, err := packs.LoadTextPack("infergo-basic-sst2")
	require.NoError(t, err)
	t.Cleanup(func() { require.NoError(t, pack.Close()) })

	server := httptest.NewServer(NewTextPackMux(pack))
	t.Cleanup(server.Close)

	resp, err := http.Get(server.URL + "/metadata")
	require.NoError(t, err)
	defer resp.Body.Close()
	require.Equal(t, http.StatusOK, resp.StatusCode)

	var metadata MetadataResponse
	require.NoError(t, json.NewDecoder(resp.Body).Decode(&metadata))
	require.Equal(t, "text-classification", metadata.Task)
	require.Equal(t, "infergo-basic-sst2", metadata.PackKey)
	require.True(t, metadata.SupportsRawText)

	body := bytes.NewBufferString(`{"text":"This product is excellent and reliable."}`)
	resp, err = http.Post(server.URL+"/predict", "application/json", body)
	require.NoError(t, err)
	defer resp.Body.Close()
	require.Equal(t, http.StatusOK, resp.StatusCode)

	var prediction TextPredictResponse
	require.NoError(t, json.NewDecoder(resp.Body).Decode(&prediction))
	require.Equal(t, "POSITIVE", prediction.ObservedLabel)
	require.NotEmpty(t, prediction.ObservedLogits)
}

func TestNewTextPackMuxMethodNotAllowed(t *testing.T) {
	pack, err := packs.LoadTextPack("infergo-basic-sst2")
	require.NoError(t, err)
	t.Cleanup(func() { require.NoError(t, pack.Close()) })

	req := httptest.NewRequest(http.MethodGet, "/predict", nil)
	rec := httptest.NewRecorder()

	NewTextPackMux(pack).ServeHTTP(rec, req)

	require.Equal(t, http.StatusMethodNotAllowed, rec.Code)
	var payload ErrorResponse
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &payload))
	require.Contains(t, payload.Error, "expected POST")
}

func TestNewTokenPackMuxPredictText(t *testing.T) {
	pack, err := packs.LoadTokenPack("infergo-basic-french-ner")
	require.NoError(t, err)
	t.Cleanup(func() { require.NoError(t, pack.Close()) })

	server := httptest.NewServer(NewTokenPackMux(pack))
	t.Cleanup(server.Close)

	body := bytes.NewBufferString(`{"text":"Sophie Tremblay a parlé avec Hydro-Québec à Montréal."}`)
	resp, err := http.Post(server.URL+"/predict", "application/json", body)
	require.NoError(t, err)
	defer resp.Body.Close()
	require.Equal(t, http.StatusOK, resp.StatusCode)

	var prediction TokenPredictResponse
	require.NoError(t, json.NewDecoder(resp.Body).Decode(&prediction))
	require.NotEmpty(t, prediction.TokenLabels)
	require.NotEmpty(t, prediction.TokenLogits)
}

func TestNewTokenPackMuxRawTextUnsupported(t *testing.T) {
	pack, err := packs.LoadTokenPack("distilbert-ner")
	require.NoError(t, err)
	t.Cleanup(func() { require.NoError(t, pack.Close()) })

	req := httptest.NewRequest(http.MethodPost, "/predict", bytes.NewBufferString(`{"text":"Angela Merkel visited Berlin."}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()

	NewTokenPackMux(pack).ServeHTTP(rec, req)

	require.Equal(t, http.StatusBadRequest, rec.Code)
	var payload ErrorResponse
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &payload))
	require.Contains(t, payload.Error, "does not support raw-text tokenization")
}
