import { GoogleGenAI } from "@google/genai";
import { CouchbaseDocument } from "../types";

// Initialize Gemini
// Note: In a real app, ensure API_KEY is handled securely (proxy/backend).
// Since this is a frontend-only demo, we expect the key in process.env or input (but input is forbidden by instructions).
// We will assume process.env.API_KEY is available or gracefully degrade if not.

let ai: GoogleGenAI | null = null;

try {
  // Check if env var exists to avoid crashing if not set in this demo environment
  if (process.env.API_KEY) {
    ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
  }
} catch (e) {
  console.warn("Gemini API Key not found. AI features will be mocked.");
}

export const GeminiService = {
  analyzeDocument: async (textContext: string): Promise<{ summary: string; tags: string[] }> => {
    if (!ai) {
      // Fallback mock if no API key
      return {
        summary: "AI processing unavailable. This is a generated placeholder summary for the uploaded document.",
        tags: ["pending-ai", "document"]
      };
    }

    try {
      const model = 'gemini-2.5-flash';
      const prompt = `
        You are an intelligent Content Management System assistant. 
        Analyze the following text extract from a document.
        
        Text: "${textContext.substring(0, 5000)}..."

        1. Provide a concise summary (max 2 sentences).
        2. Extract 3-5 relevant topic tags.
        
        Return JSON format: { "summary": string, "tags": string[] }
      `;

      const response = await ai.models.generateContent({
        model,
        contents: prompt,
        config: {
          responseMimeType: "application/json"
        }
      });

      const responseText = response.text;
      if (!responseText) throw new Error("No response from Gemini");

      return JSON.parse(responseText);

    } catch (error) {
      console.error("Gemini Analysis Failed:", error);
      return {
        summary: "Analysis failed. Please try again later.",
        tags: ["error"]
      };
    }
  }
};
