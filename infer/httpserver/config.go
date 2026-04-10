package httpserver

import (
	"log"
	"net/http"
	"time"
)

// Logger is the minimal logging contract used by the HTTP serving package.
type Logger interface {
	Printf(format string, v ...any)
}

// Config controls handler-level behavior such as logging.
type Config struct {
	Logger      Logger
	LogRequests bool
}

// Option mutates handler-level HTTP server behavior.
type Option func(*Config)

// DefaultConfig returns the default handler configuration.
func DefaultConfig() Config {
	return Config{
		Logger:      log.Default(),
		LogRequests: false,
	}
}

// WithLogger sets the logger used by the HTTP serving package.
func WithLogger(logger Logger) Option {
	return func(cfg *Config) {
		cfg.Logger = logger
	}
}

// WithRequestLogging toggles structured request logging.
func WithRequestLogging(enabled bool) Option {
	return func(cfg *Config) {
		cfg.LogRequests = enabled
	}
}

func applyOptions(options []Option) Config {
	cfg := DefaultConfig()
	for _, option := range options {
		if option != nil {
			option(&cfg)
		}
	}
	return cfg
}

// ServerConfig controls the HTTP server process behavior.
type ServerConfig struct {
	Addr              string
	ReadTimeout       time.Duration
	ReadHeaderTimeout time.Duration
	WriteTimeout      time.Duration
	IdleTimeout       time.Duration
	ShutdownTimeout   time.Duration
}

// DefaultServerConfig returns the default server configuration used by
// InferGo's first-class serving path.
func DefaultServerConfig() ServerConfig {
	return ServerConfig{
		Addr:              ":8080",
		ReadTimeout:       5 * time.Second,
		ReadHeaderTimeout: 2 * time.Second,
		WriteTimeout:      10 * time.Second,
		IdleTimeout:       30 * time.Second,
		ShutdownTimeout:   10 * time.Second,
	}
}

// NewServer creates a configured HTTP server around an InferGo handler.
func NewServer(handler http.Handler, cfg ServerConfig) *http.Server {
	if cfg.Addr == "" {
		cfg.Addr = DefaultServerConfig().Addr
	}
	return &http.Server{
		Addr:              cfg.Addr,
		Handler:           handler,
		ReadTimeout:       cfg.ReadTimeout,
		ReadHeaderTimeout: cfg.ReadHeaderTimeout,
		WriteTimeout:      cfg.WriteTimeout,
		IdleTimeout:       cfg.IdleTimeout,
	}
}
