import { mpesaConfig, mpesaEndpoints } from "./config";
import crypto from "crypto";

interface MpesaTokenResponse {
  access_token: string;
}

interface MpesaBaseResponse {
  ResponseCode: string;
  ResponseDescription: string;
}

interface MpesaError {
  message?: string;
  error?: string;
}

interface MpesaErrorResponse {
  requestId: string;
  errorCode: string;
  errorMessage: string;
}

/**
 * Generate MPESA API authentication token
 */
export async function generateMpesaToken(): Promise<string> {
  try {
    const auth = Buffer.from(
      `${mpesaConfig.consumerKey}:${mpesaConfig.consumerSecret}`,
    ).toString("base64");

    const response = await fetch(
      `${mpesaConfig.baseUrl}${mpesaEndpoints.auth}`,
      {
        method: "GET",
        headers: {
          Authorization: `Basic ${auth}`,
        },
      },
    );

    if (!response.ok) {
      const errorData = (await response
        .json()
        .catch(() => null)) as MpesaErrorResponse | null;
      console.error("MPESA Token Error:", {
        status: response.status,
        statusText: response.statusText,
        errorData,
      });
      throw new Error(`Failed to get MPESA token: ${response.statusText}`);
    }

    const data = (await response.json()) as MpesaTokenResponse;
    return data.access_token;
  } catch (error) {
    console.error("MPESA Token Generation Error:", error);
    throw error;
  }
}

/**
 * Generate MPESA API password
 */
export function generateMpesaPassword(): string {
  const timestamp = generateMpesaTimestamp();
  const str = mpesaConfig.businessShortCode + mpesaConfig.passkey + timestamp;

  // Log the components for debugging
  console.log("MPESA Password Components:", {
    shortCode: mpesaConfig.businessShortCode,
    passkey: mpesaConfig.passkey,
    timestamp,
    concatenatedString: str,
  });

  return Buffer.from(str).toString("base64");
}

/**
 * Generate MPESA API timestamp
 */
export function generateMpesaTimestamp(): string {
  const now = new Date();
  const year = now.getFullYear();
  const month = String(now.getMonth() + 1).padStart(2, "0");
  const day = String(now.getDate()).padStart(2, "0");
  const hours = String(now.getHours()).padStart(2, "0");
  const minutes = String(now.getMinutes()).padStart(2, "0");
  const seconds = String(now.getSeconds()).padStart(2, "0");

  return `${year}${month}${day}${hours}${minutes}${seconds}`;
}

/**
 * Format phone number to MPESA format (254XXXXXXXXX)
 */
export function formatPhoneNumber(phone: string): string {
  // Remove any non-digit characters
  const cleaned = phone.replace(/\D/g, "");

  // If number starts with 0, replace with 254
  if (cleaned.startsWith("0")) {
    return "254" + cleaned.slice(1);
  }

  // If number starts with +, remove it
  if (cleaned.startsWith("+")) {
    return cleaned.slice(1);
  }

  // If number is 9 digits, add 254
  if (cleaned.length === 9) {
    return "254" + cleaned;
  }

  return cleaned;
}

/**
 * Generate a unique transaction reference
 */
export function generateTransactionReference(prefix = "TRX"): string {
  const timestamp = Date.now().toString();
  const random = Math.random().toString(36).substring(2, 8).toUpperCase();
  return `${prefix}-${timestamp}-${random}`;
}

/**
 * Validate MPESA phone number format
 */
export function isValidMpesaPhone(phone: string): boolean {
  const formatted = formatPhoneNumber(phone);
  return /^254[0-9]{9}$/.test(formatted);
}

/**
 * Generate MPESA API checksum
 */
export function generateChecksum(data: string, algorithm = "sha256"): string {
  return crypto.createHash(algorithm).update(data).digest("hex");
}

/**
 * Validate MPESA API response
 */
export function validateMpesaResponse(response: unknown): boolean {
  return Boolean(
    response &&
      typeof response === "object" &&
      "ResponseCode" in response &&
      "ResponseDescription" in response,
  );
}

/**
 * Handle MPESA API errors
 */
export function handleMpesaError(error: unknown): Error {
  if (error instanceof Error) {
    return error;
  }

  if (typeof error === "string") {
    return new Error(error);
  }

  if (error && typeof error === "object") {
    const mpesaError = error as MpesaError;
    return new Error(
      mpesaError.message ??
        mpesaError.error ??
        "An unknown error occurred with MPESA API",
    );
  }

  return new Error("An unknown error occurred with MPESA API");
}
