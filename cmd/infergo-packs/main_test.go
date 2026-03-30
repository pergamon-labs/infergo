package main

import "testing"

func TestDisplayPath(t *testing.T) {
	t.Run("relative to cwd", func(t *testing.T) {
		got := displayPath("/repo", "/repo/testdata/reference.json")
		if got != "testdata/reference.json" {
			t.Fatalf("displayPath() = %q, want %q", got, "testdata/reference.json")
		}
	})

	t.Run("outside cwd falls back to relative traversal", func(t *testing.T) {
		got := displayPath("/repo", "/tmp/reference.json")
		if got != "../tmp/reference.json" {
			t.Fatalf("displayPath() = %q, want %q", got, "../tmp/reference.json")
		}
	})

	t.Run("relative input stays normalized", func(t *testing.T) {
		got := displayPath("/repo", "testdata/native/../reference.json")
		if got != "testdata/reference.json" {
			t.Fatalf("displayPath() = %q, want %q", got, "testdata/reference.json")
		}
	})
}
