import { generateObject, generateText } from "ai";
import { openrouter } from "./services";
import { ActivityTracker, ModelCallOptions, ResearchState } from "./types";
import { MAX_RETRY_ATTEMPTS, RETRY_DELAY_MS } from "./constants";
import { delay } from "./utils";

export async function callModel<T>(
  {
    model,
    prompt,
    system,
    schema,
    activityType = "generate",
  }: ModelCallOptions<T>,
  researchState: ResearchState,
  activityTracker: ActivityTracker
): Promise<T | string> {
  let attempts = 0;
  let lastError: Error | null = null;

  // Log the start of the model call
  console.log(`[Model Call] Starting call to model: ${model}`);
  console.log(`[Model Call] Activity Type: ${activityType}`);
  console.log(`[Model Call] System Prompt Length: ${system?.length || 0}`);
  console.log(`[Model Call] User Prompt Length: ${prompt?.length || 0}`);

  while (attempts < MAX_RETRY_ATTEMPTS) {
    try {
      if (schema) {
        console.log(
          `[Model Call] Attempt ${
            attempts + 1
          }/${MAX_RETRY_ATTEMPTS} - Generating object with schema`
        );
        const { object, usage } = await generateObject({
          model: openrouter(model),
          prompt,
          system,
          schema: schema,
        });

        researchState.tokenUsed += usage.totalTokens;
        researchState.completedSteps++;

        console.log(`[Model Call] Success - Tokens used: ${usage.totalTokens}`);
        return object;
      } else {
        console.log(
          `[Model Call] Attempt ${
            attempts + 1
          }/${MAX_RETRY_ATTEMPTS} - Generating text`
        );
        const { text, usage } = await generateText({
          model: openrouter(model),
          prompt,
          system,
        });

        researchState.tokenUsed += usage.totalTokens;
        researchState.completedSteps++;

        console.log(`[Model Call] Success - Tokens used: ${usage.totalTokens}`);
        return text;
      }
    } catch (error) {
      attempts++;
      lastError = error instanceof Error ? error : new Error("Unknown error");

      // Enhanced error logging
      console.error(
        `[Model Call] Error on attempt ${attempts}/${MAX_RETRY_ATTEMPTS}:`,
        {
          error: lastError.message,
          stack: lastError.stack,
          model,
          activityType,
          timestamp: new Date().toISOString(),
        }
      );

      if (attempts < MAX_RETRY_ATTEMPTS) {
        activityTracker.add(
          activityType,
          "warning",
          `Model call failed, attempt ${attempts}/${MAX_RETRY_ATTEMPTS}. Retrying...`
        );
        console.log(
          `[Model Call] Waiting ${RETRY_DELAY_MS * attempts}ms before retry...`
        );
        await delay(RETRY_DELAY_MS * attempts);
      }
    }
  }

  // Final error logging before throwing
  console.error(`[Model Call] Failed after ${MAX_RETRY_ATTEMPTS} attempts:`, {
    lastError: lastError?.message,
    model,
    activityType,
    totalAttempts: attempts,
    timestamp: new Date().toISOString(),
  });

  throw lastError || new Error(`Failed after ${MAX_RETRY_ATTEMPTS} attempts!`);
}
