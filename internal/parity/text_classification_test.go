package parity

import "testing"

func TestRunTextClassificationFixture(t *testing.T) {
	t.Parallel()

	report, err := RunTextClassificationFixture("../../testdata/parity/text-classification/fixture.json")
	if err != nil {
		t.Fatalf("RunTextClassificationFixture() error = %v", err)
	}

	if !report.Passed() {
		t.Fatalf("expected parity fixture to pass, got report:\n%s", report.String())
	}

	if len(report.CaseResults) != 4 {
		t.Fatalf("expected 4 parity cases, got %d", len(report.CaseResults))
	}
}
