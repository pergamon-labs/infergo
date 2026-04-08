package main

import (
	"context"
	"errors"
	"flag"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/pergamon-labs/infergo/infer/experimental/entres"
	"github.com/pergamon-labs/infergo/infer/httpserver"
)

func main() {
	serverCfg := httpserver.DefaultServerConfig()

	addr := flag.String("addr", envString("INFERGO_ENTRES_SERVE_ADDR", serverCfg.Addr), "http listen address")
	bundleDir := flag.String("bundle", envString("INFERGO_ENTRES_SERVE_BUNDLE", ""), "family-2 entres bridge bundle directory")
	readTimeout := flag.Duration("read-timeout", envDuration("INFERGO_ENTRES_SERVE_READ_TIMEOUT", serverCfg.ReadTimeout), "http read timeout")
	readHeaderTimeout := flag.Duration("read-header-timeout", envDuration("INFERGO_ENTRES_SERVE_READ_HEADER_TIMEOUT", serverCfg.ReadHeaderTimeout), "http read-header timeout")
	writeTimeout := flag.Duration("write-timeout", envDuration("INFERGO_ENTRES_SERVE_WRITE_TIMEOUT", serverCfg.WriteTimeout), "http write timeout")
	idleTimeout := flag.Duration("idle-timeout", envDuration("INFERGO_ENTRES_SERVE_IDLE_TIMEOUT", serverCfg.IdleTimeout), "http idle timeout")
	shutdownTimeout := flag.Duration("shutdown-timeout", envDuration("INFERGO_ENTRES_SERVE_SHUTDOWN_TIMEOUT", serverCfg.ShutdownTimeout), "graceful shutdown timeout")
	flag.Parse()

	if strings.TrimSpace(*bundleDir) == "" {
		log.Fatal("bundle path is required")
	}

	serverCfg.Addr = *addr
	serverCfg.ReadTimeout = *readTimeout
	serverCfg.ReadHeaderTimeout = *readHeaderTimeout
	serverCfg.WriteTimeout = *writeTimeout
	serverCfg.IdleTimeout = *idleTimeout
	serverCfg.ShutdownTimeout = *shutdownTimeout

	model, err := entres.Load(*bundleDir)
	if err != nil {
		log.Fatalf("load entres bridge bundle: %v", err)
	}
	defer model.Close()

	mux := entres.NewMux(model)
	serve(serverCfg, mux, model)
}

func serve(cfg httpserver.ServerConfig, handler http.Handler, model *entres.Model) {
	server := httpserver.NewServer(handler, cfg)
	logServeHints(cfg, model)

	errCh := make(chan error, 1)
	go func() {
		errCh <- server.ListenAndServe()
	}()

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	select {
	case err := <-errCh:
		if err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Fatal(err)
		}
	case <-ctx.Done():
		log.Printf("InferGo entres bridge shutting down after signal: %v", ctx.Err())
		shutdownCtx, cancel := context.WithTimeout(context.Background(), cfg.ShutdownTimeout)
		defer cancel()
		if err := server.Shutdown(shutdownCtx); err != nil {
			log.Fatalf("graceful shutdown failed: %v", err)
		}
		if err := <-errCh; err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Fatal(err)
		}
	}
}

func logServeHints(cfg httpserver.ServerConfig, model *entres.Model) {
	addr := cfg.Addr
	curlAddr := addr
	if strings.HasPrefix(curlAddr, ":") {
		curlAddr = "127.0.0.1" + curlAddr
	}
	meta := model.Metadata()

	log.Printf("InferGo experimental entres bridge serving %q on %s", model.ModelID(), addr)
	log.Printf("Family=%s task=%s profile_kind=%s vector_size=%d message_size=%d", meta.Family, meta.Task, meta.ProfileKind, meta.VectorSize, meta.MessageSize)
	log.Printf("Timeouts: read=%s read-header=%s write=%s idle=%s shutdown=%s", cfg.ReadTimeout, cfg.ReadHeaderTimeout, cfg.WriteTimeout, cfg.IdleTimeout, cfg.ShutdownTimeout)
	log.Printf("Health: curl -s http://%s/healthz | jq", curlAddr)
	log.Printf("Metadata: curl -s http://%s/metadata | jq", curlAddr)
	log.Printf("Predict: curl -s -X POST http://%s/predict -H 'Content-Type: application/json' -d '{\"vectors\":[[0,0],[1,1]],\"message\":[0,0]}' | jq", curlAddr)
}

func envString(key, fallback string) string {
	if value := strings.TrimSpace(os.Getenv(key)); value != "" {
		return value
	}
	return fallback
}

func envDuration(key string, fallback time.Duration) time.Duration {
	value := strings.TrimSpace(os.Getenv(key))
	if value == "" {
		return fallback
	}
	parsed, err := time.ParseDuration(value)
	if err != nil {
		log.Fatalf("parse %s as duration: %v", key, err)
	}
	return parsed
}
