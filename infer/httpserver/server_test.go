package httpserver

import (
	"bytes"
	"encoding/json"
	"log"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/pergamon-labs/infergo/backends/bionet"
	"github.com/pergamon-labs/infergo/infer"
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
	require.Contains(t, metadata.SupportedInputs, "text")

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

func TestNewTextClassifierMuxPredictTokenizedInput(t *testing.T) {
	classifier, err := infer.LoadTextClassifier("../../testdata/native/text-classification/distilbert-sst2-embedding-masked-avg-pool")
	require.NoError(t, err)
	t.Cleanup(func() { require.NoError(t, classifier.Close()) })

	info, err := bionet.InspectTextClassificationBundle("../../testdata/native/text-classification/distilbert-sst2-embedding-masked-avg-pool")
	require.NoError(t, err)

	server := httptest.NewServer(NewTextClassifierMux(classifier, TextClassifierMetadata{
		ModelID:                info.ModelID,
		SupportsRawText:        info.SupportsRawText,
		SupportsPairText:       info.SupportsPairText,
		SupportsTokenizedInput: info.SupportsTokenizedInput,
	}))
	t.Cleanup(server.Close)

	resp, err := http.Get(server.URL + "/metadata")
	require.NoError(t, err)
	defer resp.Body.Close()
	require.Equal(t, http.StatusOK, resp.StatusCode)

	var metadata MetadataResponse
	require.NoError(t, json.NewDecoder(resp.Body).Decode(&metadata))
	require.False(t, metadata.SupportsRawText)
	require.True(t, metadata.SupportsTokenizedInput)
	require.Contains(t, metadata.SupportedInputs, "input_ids")

	body := bytes.NewBufferString(`{"input_ids":[101,2023,2003,1037,2742,102],"attention_mask":[1,1,1,1,1,1]}`)
	resp, err = http.Post(server.URL+"/predict", "application/json", body)
	require.NoError(t, err)
	defer resp.Body.Close()
	require.Equal(t, http.StatusOK, resp.StatusCode)

	var prediction TextPredictResponse
	require.NoError(t, json.NewDecoder(resp.Body).Decode(&prediction))
	require.NotEmpty(t, prediction.ObservedLabel)
	require.NotEmpty(t, prediction.ObservedLogits)
}

func TestNewTextClassifierMuxPredictRawText(t *testing.T) {
	classifier := &stubTextClassifier{
		prediction: infer.TextPrediction{
			Backend: "bionet",
			ModelID: "example/model",
			Labels:  []string{"NEGATIVE", "POSITIVE"},
			Logits:  []float64{-0.5, 0.8},
			Label:   "POSITIVE",
		},
	}

	server := httptest.NewServer(NewTextClassifierMux(classifier, TextClassifierMetadata{
		ModelID:                "example/model",
		SupportsRawText:        true,
		SupportsTokenizedInput: true,
		RawTextEncoder: func(text, textPair string) (infer.TextInput, error) {
			require.Equal(t, "This product is excellent and reliable.", text)
			require.Empty(t, textPair)
			return infer.TextInput{
				InputIDs:      []int64{101, 2023, 4031, 2003, 6581, 1998, 10539, 1012, 102},
				AttentionMask: []int64{1, 1, 1, 1, 1, 1, 1, 1, 1},
			}, nil
		},
	}))
	t.Cleanup(server.Close)

	body := bytes.NewBufferString(`{"text":"This product is excellent and reliable."}`)
	resp, err := http.Post(server.URL+"/predict", "application/json", body)
	require.NoError(t, err)
	defer resp.Body.Close()
	require.Equal(t, http.StatusOK, resp.StatusCode)

	require.Equal(t, []infer.TextInput{{
		InputIDs:      []int64{101, 2023, 4031, 2003, 6581, 1998, 10539, 1012, 102},
		AttentionMask: []int64{1, 1, 1, 1, 1, 1, 1, 1, 1},
	}}, classifier.inputs)
}

func TestNewTextClassifierMuxPredictPairText(t *testing.T) {
	classifier := &stubTextClassifier{
		prediction: infer.TextPrediction{
			Backend: "bionet",
			ModelID: "example/mrpc",
			Labels:  []string{"LABEL_0", "LABEL_1"},
			Logits:  []float64{0.7, -0.2},
			Label:   "LABEL_0",
		},
	}

	server := httptest.NewServer(NewTextClassifierMux(classifier, TextClassifierMetadata{
		ModelID:                "example/mrpc",
		SupportsRawText:        true,
		SupportsPairText:       true,
		SupportsTokenizedInput: true,
		RawTextEncoder: func(text, textPair string) (infer.TextInput, error) {
			require.Equal(t, "The company said the deal closed.", text)
			require.Equal(t, "The acquisition has been completed, the company said.", textPair)
			return infer.TextInput{
				InputIDs:      []int64{101, 207, 208, 209, 207, 210, 211, 206, 102, 207, 212, 213, 214, 215, 216, 207, 208, 102},
				AttentionMask: []int64{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
			}, nil
		},
	}))
	t.Cleanup(server.Close)

	body := bytes.NewBufferString(`{"text":"The company said the deal closed.","text_pair":"The acquisition has been completed, the company said."}`)
	resp, err := http.Post(server.URL+"/predict", "application/json", body)
	require.NoError(t, err)
	defer resp.Body.Close()
	require.Equal(t, http.StatusOK, resp.StatusCode)
}

func TestNewTextClassifierMuxRejectsUnsupportedInputMode(t *testing.T) {
	classifier, err := infer.LoadTextClassifier("../../testdata/native/text-classification/distilbert-sst2-embedding-masked-avg-pool")
	require.NoError(t, err)
	t.Cleanup(func() { require.NoError(t, classifier.Close()) })

	req := httptest.NewRequest(http.MethodPost, "/predict", bytes.NewBufferString(`{"text":"hello world"}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()

	NewTextClassifierMux(classifier, TextClassifierMetadata{
		ModelID:                "example/model",
		SupportsTokenizedInput: true,
	}).ServeHTTP(rec, req)

	require.Equal(t, http.StatusBadRequest, rec.Code)
	var payload ErrorResponse
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &payload))
	require.Equal(t, "invalid_request", payload.Error.Code)
	require.Contains(t, payload.Error.Message, "tokenized input only")
}

func TestNewTextClassifierMuxRejectsPairTextWhenUnsupported(t *testing.T) {
	classifier := &stubTextClassifier{}

	req := httptest.NewRequest(http.MethodPost, "/predict", bytes.NewBufferString(`{"text":"The company said the deal closed.","text_pair":"The acquisition has been completed."}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()

	NewTextClassifierMux(classifier, TextClassifierMetadata{
		ModelID:                "example/model",
		SupportsRawText:        true,
		SupportsPairText:       false,
		SupportsTokenizedInput: true,
		RawTextEncoder: func(text, textPair string) (infer.TextInput, error) {
			return infer.TextInput{}, nil
		},
	}).ServeHTTP(rec, req)

	require.Equal(t, http.StatusBadRequest, rec.Code)
	var payload ErrorResponse
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &payload))
	require.Equal(t, "invalid_request", payload.Error.Code)
	require.Contains(t, payload.Error.Message, "does not support paired text input")
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
	require.Equal(t, "method_not_allowed", payload.Error.Code)
	require.Contains(t, payload.Error.Message, "expected POST")
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
	require.Equal(t, "invalid_request", payload.Error.Code)
	require.Contains(t, payload.Error.Message, "does not support raw-text tokenization")
}

func TestNewTextPackMuxRejectsMultipleInputs(t *testing.T) {
	pack, err := packs.LoadTextPack("infergo-basic-sst2")
	require.NoError(t, err)
	t.Cleanup(func() { require.NoError(t, pack.Close()) })

	req := httptest.NewRequest(http.MethodPost, "/predict", bytes.NewBufferString(`{"text":"good","tokens":["good"]}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()

	NewTextPackMux(pack).ServeHTTP(rec, req)

	require.Equal(t, http.StatusBadRequest, rec.Code)
	var payload ErrorResponse
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &payload))
	require.Equal(t, "invalid_request", payload.Error.Code)
	require.Contains(t, payload.Error.Message, "exactly one")
}

func TestNewTextPackMuxNotFoundIsJSON(t *testing.T) {
	pack, err := packs.LoadTextPack("infergo-basic-sst2")
	require.NoError(t, err)
	t.Cleanup(func() { require.NoError(t, pack.Close()) })

	req := httptest.NewRequest(http.MethodGet, "/unknown", nil)
	rec := httptest.NewRecorder()

	NewTextPackMux(pack).ServeHTTP(rec, req)

	require.Equal(t, http.StatusNotFound, rec.Code)
	var payload ErrorResponse
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &payload))
	require.Equal(t, "not_found", payload.Error.Code)
}

func TestNewTextPackMuxRequestLogging(t *testing.T) {
	pack, err := packs.LoadTextPack("infergo-basic-sst2")
	require.NoError(t, err)
	t.Cleanup(func() { require.NoError(t, pack.Close()) })

	var buf bytes.Buffer
	logger := log.New(&buf, "", 0)

	req := httptest.NewRequest(http.MethodGet, "/metadata", nil)
	rec := httptest.NewRecorder()

	NewTextPackMux(pack, WithLogger(logger), WithRequestLogging(true)).ServeHTTP(rec, req)

	require.Equal(t, http.StatusOK, rec.Code)
	require.Contains(t, buf.String(), "route=metadata")
	require.Contains(t, buf.String(), "status=200")
}

type stubTextClassifier struct {
	inputs     []infer.TextInput
	prediction infer.TextPrediction
	predictErr error
	closeErr   error
}

func (s *stubTextClassifier) BackendName() string {
	return "bionet"
}

func (s *stubTextClassifier) ModelID() string {
	return s.prediction.ModelID
}

func (s *stubTextClassifier) Labels() []string {
	return s.prediction.Labels
}

func (s *stubTextClassifier) Predict(input infer.TextInput) (infer.TextPrediction, error) {
	s.inputs = append(s.inputs, input)
	return s.prediction, s.predictErr
}

func (s *stubTextClassifier) PredictBatch(inputs []infer.TextInput) ([]infer.TextPrediction, error) {
	s.inputs = append(s.inputs, inputs...)
	if s.predictErr != nil {
		return nil, s.predictErr
	}
	output := make([]infer.TextPrediction, len(inputs))
	for i := range output {
		output[i] = s.prediction
	}
	return output, nil
}

func (s *stubTextClassifier) Close() error {
	return s.closeErr
}
