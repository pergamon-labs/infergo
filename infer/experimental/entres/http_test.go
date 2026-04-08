package entres

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

type fakeScorer struct {
	meta Metadata
}

func (f fakeScorer) BackendName() string { return f.meta.Backend }
func (f fakeScorer) ModelID() string     { return f.meta.ModelID }
func (f fakeScorer) Metadata() Metadata  { return f.meta }
func (f fakeScorer) Close() error        { return nil }
func (f fakeScorer) Predict(input Input) (Prediction, error) {
	return Prediction{
		Backend: f.meta.Backend,
		ModelID: f.meta.ModelID,
		Scores:  []float64{0.25, 0.75},
	}, nil
}

func TestNewMuxMetadata(t *testing.T) {
	t.Parallel()

	mux := NewMux(fakeScorer{meta: Metadata{
		Family:      "numeric-feature-scoring",
		Task:        "entity-resolution-scoring",
		Backend:     "torchscript",
		ModelID:     "pergamon/entres-individual",
		ProfileKind: "individual",
		VectorSize:  128,
		MessageSize: 128,
	}})

	req := httptest.NewRequest(http.MethodGet, "/metadata", nil)
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("unexpected status %d", rec.Code)
	}
	if got := rec.Body.String(); !bytes.Contains([]byte(got), []byte(`"family":"numeric-feature-scoring"`)) {
		t.Fatalf("unexpected body %s", got)
	}
}

func TestNewMuxPredict(t *testing.T) {
	t.Parallel()

	mux := NewMux(fakeScorer{meta: Metadata{
		Family:      "numeric-feature-scoring",
		Task:        "entity-resolution-scoring",
		Backend:     "torchscript",
		ModelID:     "pergamon/entres-individual",
		VectorSize:  2,
		MessageSize: 2,
	}})

	body, err := json.Marshal(HTTPPredictRequest{
		Vectors: [][]float64{{1, 2}, {3, 4}},
		Message: []float64{5, 6},
	})
	if err != nil {
		t.Fatalf("Marshal() error = %v", err)
	}

	req := httptest.NewRequest(http.MethodPost, "/predict", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("unexpected status %d body=%s", rec.Code, rec.Body.String())
	}
	if got := rec.Body.String(); !bytes.Contains([]byte(got), []byte(`"scores":[0.25,0.75]`)) {
		t.Fatalf("unexpected body %s", got)
	}
}
