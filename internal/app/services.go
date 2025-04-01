package app

import (
	"context"
	"database/sql"

	"github.com/kujtimiihoxha/termai/internal/config"
	"github.com/kujtimiihoxha/termai/internal/db"
	"github.com/kujtimiihoxha/termai/internal/logging"
	"github.com/kujtimiihoxha/termai/internal/message"
	"github.com/kujtimiihoxha/termai/internal/permission"
	"github.com/kujtimiihoxha/termai/internal/session"
)

type App struct {
	Context context.Context

	Sessions    session.Service
	Messages    message.Service
	Permissions permission.Service

	Logger logging.Interface
}

func New(ctx context.Context, conn *sql.DB) *App {
	q := db.New(conn)
	log := logging.NewLogger(logging.Options{
		Level: config.Get().Log.Level,
	})
	sessions := session.NewService(ctx, q)
	messages := message.NewService(ctx, q)

	return &App{
		Context:     ctx,
		Sessions:    sessions,
		Messages:    messages,
		Permissions: permission.Default,
		Logger:      log,
	}
}

