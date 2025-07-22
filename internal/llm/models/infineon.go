package models

const (
	ProviderInfineon ModelProvider = "infineon"

	// Infineon-spezifische Modell-IDs
	InfineonGPT4o ModelID = "infineon-gpt4o"
)

var InfineonModels = map[ModelID]Model{
	InfineonGPT4o: {
		ID:                  InfineonGPT4o,
		Name:                "Infineon GPT-4o",
		Provider:            ProviderInfineon,
		APIModel:            "gpt-4o",
		CostPer1MIn:         2.50,
		CostPer1MInCached:   1.25,
		CostPer1MOutCached:  0.0,
		CostPer1MOut:        10.00,
		ContextWindow:       128_000,
		DefaultMaxTokens:    4096,
		SupportsAttachments: true,
	},
} 