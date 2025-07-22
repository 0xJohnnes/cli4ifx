package provider

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"time"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/shared"
	"https://github.com/0xJohnnes/cli4ifx/internal/config"
	"https://github.com/0xJohnnes/cli4ifx/internal/llm/models"
	"https://github.com/0xJohnnes/cli4ifx/internal/llm/tools"
	"https://github.com/0xJohnnes/cli4ifx/internal/logging"
	"https://github.com/0xJohnnes/cli4ifx/internal/message"
)

type infineonOptions struct {
	baseURL         string
	disableCache    bool
	extraHeaders    map[string]string
}

type InfineonOption func(*infineonOptions)

type infineonClient struct {
	providerOptions providerClientOptions
	options         infineonOptions
	client          openai.Client
}

type InfineonClient ProviderClient

func newInfineonClient(opts providerClientOptions) InfineonClient {
	infineonOpts := infineonOptions{
		baseURL: "https://api.infineon.ai/v1", // Standard-URL für den Infineon-API-Endpunkt
	}
	for _, o := range opts.infineonOptions {
		o(&infineonOpts)
	}

	openaiClientOptions := []option.RequestOption{}
	if opts.apiKey != "" {
		openaiClientOptions = append(openaiClientOptions, option.WithAPIKey(opts.apiKey))
	}
	if infineonOpts.baseURL != "" {
		openaiClientOptions = append(openaiClientOptions, option.WithBaseURL(infineonOpts.baseURL))
	}

	if infineonOpts.extraHeaders != nil {
		for key, value := range infineonOpts.extraHeaders {
			openaiClientOptions = append(openaiClientOptions, option.WithHeader(key, value))
		}
	}

	client := openai.NewClient(openaiClientOptions...)
	return &infineonClient{
		providerOptions: opts,
		options:         infineonOpts,
		client:          client,
	}
}

func (i *infineonClient) convertMessages(messages []message.Message) (openaiMessages []openai.ChatCompletionMessageParamUnion) {
	// Add system message first
	openaiMessages = append(openaiMessages, openai.SystemMessage(i.providerOptions.systemMessage))

	for _, msg := range messages {
		switch msg.Role {
		case message.User:
			var content []openai.ChatCompletionContentPartUnionParam
			textBlock := openai.ChatCompletionContentPartTextParam{Text: msg.Content().String()}
			content = append(content, openai.ChatCompletionContentPartUnionParam{OfText: &textBlock})
			for _, binaryContent := range msg.BinaryContent() {
				imageURL := openai.ChatCompletionContentPartImageImageURLParam{URL: binaryContent.String(models.ProviderInfineon)}
				imageBlock := openai.ChatCompletionContentPartImageParam{ImageURL: imageURL}

				content = append(content, openai.ChatCompletionContentPartUnionParam{OfImageURL: &imageBlock})
			}

			openaiMessages = append(openaiMessages, openai.UserMessage(content))

		case message.Assistant:
			assistantMsg := openai.ChatCompletionAssistantMessageParam{
				Role: "assistant",
			}

			if msg.Content().String() != "" {
				assistantMsg.Content = openai.ChatCompletionAssistantMessageParamContentUnion{
					OfString: openai.String(msg.Content().String()),
				}
			}

			if len(msg.ToolCalls()) > 0 {
				assistantMsg.ToolCalls = make([]openai.ChatCompletionMessageToolCallParam, len(msg.ToolCalls()))
				for i, call := range msg.ToolCalls() {
					var args map[string]interface{}
					if err := json.Unmarshal([]byte(call.Input), &args); err != nil {
						logging.Error("Failed to parse tool call input", "error", err)
						continue
					}

					argsJSON, err := json.Marshal(args)
					if err != nil {
						logging.Error("Failed to marshal tool call args", "error", err)
						continue
					}

					assistantMsg.ToolCalls[i] = openai.ChatCompletionMessageToolCallParam{
						ID:       call.ID,
						Type:     "function",
						Function: openai.ChatCompletionMessageToolCallFunctionParam{Name: call.Name, Arguments: string(argsJSON)},
					}
				}
			}

			openaiMessages = append(openaiMessages, openai.ChatCompletionMessageParamUnion{OfAssistant: &assistantMsg})

		case message.Tool:
			for _, result := range msg.ToolResults() {
				toolMsg := openai.ChatCompletionToolMessageParam{
					Role:      "tool",
					Content:   result.Content,
					ToolCallID: result.CallID,
				}
				openaiMessages = append(openaiMessages, openai.ChatCompletionMessageParamUnion{OfTool: &toolMsg})
			}
		}
	}
	return
}

func (i *infineonClient) convertTools(tools []tools.BaseTool) []openai.ChatCompletionToolParam {
	if len(tools) == 0 {
		return nil
	}

	openaiTools := make([]openai.ChatCompletionToolParam, len(tools))
	for idx, tool := range tools {
		schema, err := tool.JSONSchema()
		if err != nil {
			logging.Error("Failed to get tool schema", "error", err)
			continue
		}

		openaiTools[idx] = openai.ChatCompletionToolParam{
			Type: "function",
			Function: openai.ChatCompletionFunctionDefinitionParam{
				Name:        tool.Name(),
				Description: openai.String(tool.Description()),
				Parameters:  schema,
			},
		}
	}
	return openaiTools
}

func (i *infineonClient) finishReason(reason string) message.FinishReason {
	switch reason {
	case "stop":
		return message.FinishReasonStop
	case "length":
		return message.FinishReasonLength
	case "tool_calls":
		return message.FinishReasonToolCalls
	case "content_filter":
		return message.FinishReasonContentFilter
	default:
		return message.FinishReasonUnknown
	}
}

func (i *infineonClient) preparedParams(messages []openai.ChatCompletionMessageParamUnion, tools []openai.ChatCompletionToolParam) openai.ChatCompletionNewParams {
	params := openai.ChatCompletionNewParams{
		Messages: messages,
		Model:    i.providerOptions.model.APIModel,
	}

	if len(tools) > 0 {
		params.Tools = tools
	}

	if i.providerOptions.maxTokens > 0 {
		params.MaxTokens = openai.Int(int(i.providerOptions.maxTokens))
	}

	return params
}

func (i *infineonClient) send(ctx context.Context, messages []message.Message, tools []tools.BaseTool) (response *ProviderResponse, err error) {
	openaiMessages := i.convertMessages(messages)
	openaiTools := i.convertTools(tools)
	params := i.preparedParams(openaiMessages, openaiTools)

	var completion openai.ChatCompletion
	for attempts := 0; attempts < maxRetries; attempts++ {
		completion, err = i.client.CreateChatCompletion(ctx, params)
		if err == nil {
			break
		}

		shouldRetry, sleepDuration, retryErr := i.shouldRetry(attempts, err)
		if retryErr != nil {
			return nil, retryErr
		}
		if !shouldRetry {
			return nil, err
		}

		logging.Info("Retrying request", "attempt", attempts+1, "sleep", sleepDuration)
		time.Sleep(sleepDuration)
	}

	if err != nil {
		return nil, err
	}

	if len(completion.Choices) == 0 {
		return nil, errors.New("no choices returned")
	}

	choice := completion.Choices[0]
	content := ""
	if choice.Message.Content != nil {
		content = *choice.Message.Content
	}

	return &ProviderResponse{
		Content:      content,
		ToolCalls:    i.toolCalls(completion),
		Usage:        i.usage(completion),
		FinishReason: i.finishReason(choice.FinishReason),
	}, nil
}

func (i *infineonClient) stream(ctx context.Context, messages []message.Message, tools []tools.BaseTool) <-chan ProviderEvent {
	openaiMessages := i.convertMessages(messages)
	openaiTools := i.convertTools(tools)
	params := i.preparedParams(openaiMessages, openaiTools)
	params.Stream = openai.Bool(true)

	eventChan := make(chan ProviderEvent)

	go func() {
		defer close(eventChan)

		var stream openai.ChatCompletionStream
		var err error

		for attempts := 0; attempts < maxRetries; attempts++ {
			stream, err = i.client.CreateChatCompletionStream(ctx, params)
			if err == nil {
				break
			}

			shouldRetry, sleepDuration, retryErr := i.shouldRetry(attempts, err)
			if retryErr != nil {
				eventChan <- ProviderEvent{
					Type:  EventError,
					Error: retryErr,
				}
				return
			}
			if !shouldRetry {
				eventChan <- ProviderEvent{
					Type:  EventError,
					Error: err,
				}
				return
			}

			logging.Info("Retrying stream request", "attempt", attempts+1, "sleep", sleepDuration)
			time.Sleep(sleepDuration)
		}

		if err != nil {
			eventChan <- ProviderEvent{
				Type:  EventError,
				Error: err,
			}
			return
		}

		defer stream.Close()

		eventChan <- ProviderEvent{
			Type: EventContentStart,
		}

		var content string
		var toolCalls []message.ToolCall
		var usage TokenUsage
		var finishReason message.FinishReason

		for {
			chunk, err := stream.Recv()
			if errors.Is(err, io.EOF) {
				break
			}
			if err != nil {
				eventChan <- ProviderEvent{
					Type:  EventError,
					Error: err,
				}
				return
			}

			if len(chunk.Choices) == 0 {
				continue
			}

			choice := chunk.Choices[0]

			// Handle content delta
			if choice.Delta.Content != nil {
				content += *choice.Delta.Content
				eventChan <- ProviderEvent{
					Type:    EventContentDelta,
					Content: *choice.Delta.Content,
				}
			}

			// Handle tool calls
			if len(choice.Delta.ToolCalls) > 0 {
				for _, toolCallDelta := range choice.Delta.ToolCalls {
					if toolCallDelta.Index < 0 || toolCallDelta.Index >= len(toolCalls) {
						// New tool call
						if toolCallDelta.ID != nil {
							toolCall := message.ToolCall{
								ID:   *toolCallDelta.ID,
								Name: "",
							}
							toolCalls = append(toolCalls, toolCall)

							eventChan <- ProviderEvent{
								Type:     EventToolUseStart,
								ToolCall: &toolCall,
							}
						}
					}

					if toolCallDelta.Index >= 0 && toolCallDelta.Index < len(toolCalls) {
						// Update existing tool call
						if toolCallDelta.Function.Name != nil {
							toolCalls[toolCallDelta.Index].Name = *toolCallDelta.Function.Name
						}

						if toolCallDelta.Function.Arguments != nil {
							toolCalls[toolCallDelta.Index].Input += *toolCallDelta.Function.Arguments
							eventChan <- ProviderEvent{
								Type: EventToolUseDelta,
								ToolCall: &message.ToolCall{
									ID:    toolCalls[toolCallDelta.Index].ID,
									Name:  toolCalls[toolCallDelta.Index].Name,
									Input: *toolCallDelta.Function.Arguments,
								},
							}
						}
					}
				}
			}

			if choice.FinishReason != "" {
				finishReason = i.finishReason(choice.FinishReason)
			}
		}

		// Send tool use stop events for all tool calls
		for _, toolCall := range toolCalls {
			eventChan <- ProviderEvent{
				Type:     EventToolUseStop,
				ToolCall: &toolCall,
			}
		}

		eventChan <- ProviderEvent{
			Type: EventContentStop,
		}

		eventChan <- ProviderEvent{
			Type: EventComplete,
			Response: &ProviderResponse{
				Content:      content,
				ToolCalls:    toolCalls,
				Usage:        usage,
				FinishReason: finishReason,
			},
		}
	}()

	return eventChan
}

func (i *infineonClient) shouldRetry(attempts int, err error) (bool, int64, error) {
	var apiErr *shared.APIError
	if !errors.As(err, &apiErr) {
		return false, 0, nil
	}

	// Rate limit errors
	if apiErr.StatusCode == 429 {
		retryAfterHeader := apiErr.Response.Header.Get("Retry-After")
		if retryAfterHeader != "" {
			retryAfter, parseErr := time.ParseDuration(retryAfterHeader + "s")
			if parseErr == nil {
				return true, retryAfter.Milliseconds(), nil
			}
		}

		// Exponential backoff
		backoff := int64(1000 * (1 << attempts))
		return true, backoff, nil
	}

	// Server errors
	if apiErr.StatusCode >= 500 && apiErr.StatusCode < 600 {
		if attempts >= maxRetries-1 {
			return false, 0, fmt.Errorf("max retries reached: %w", err)
		}
		backoff := int64(1000 * (1 << attempts))
		return true, backoff, nil
	}

	return false, 0, nil
}

func (i *infineonClient) toolCalls(completion openai.ChatCompletion) []message.ToolCall {
	if len(completion.Choices) == 0 || len(completion.Choices[0].Message.ToolCalls) == 0 {
		return nil
	}

	toolCalls := make([]message.ToolCall, len(completion.Choices[0].Message.ToolCalls))
	for idx, call := range completion.Choices[0].Message.ToolCalls {
		toolCalls[idx] = message.ToolCall{
			ID:    call.ID,
			Name:  call.Function.Name,
			Input: call.Function.Arguments,
		}
	}
	return toolCalls
}

func (i *infineonClient) usage(completion openai.ChatCompletion) TokenUsage {
	return TokenUsage{
		InputTokens:  int64(completion.Usage.PromptTokens),
		OutputTokens: int64(completion.Usage.CompletionTokens),
	}
}

func WithInfineonBaseURL(baseURL string) InfineonOption {
	return func(options *infineonOptions) {
		options.baseURL = baseURL
	}
}

func WithInfineonExtraHeaders(headers map[string]string) InfineonOption {
	return func(options *infineonOptions) {
		options.extraHeaders = headers
	}
}

func WithInfineonDisableCache() InfineonOption {
	return func(options *infineonOptions) {
		options.disableCache = true
	}
} 