package main

import (
	"https://github.com/0xJohnnes/cli4ifx/cmd"
	"https://github.com/0xJohnnes/cli4ifx/internal/logging"
)

func main() {
	defer logging.RecoverPanic("main", func() {
		logging.ErrorPersist("Application terminated due to unhandled panic")
	})

	cmd.Execute()
}
