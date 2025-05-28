import { env } from "../../env";

export const mpesaConfig = {
  // API URLs
  baseUrl:
    env.MPESA_ENV === "production"
      ? "https://api.safaricom.co.ke"
      : "https://sandbox.safaricom.co.ke",

  // Business Details
  businessShortCode: env.MPESA_BUSINESS_SHORTCODE,
  passkey: env.MPESA_PASSKEY,

  // API Credentials
  consumerKey: env.MPESA_CONSUMER_KEY,
  consumerSecret: env.MPESA_CONSUMER_SECRET,

  // Callback URLs
  callbackUrl: env.MPESA_CALLBACK_URL,
  timeoutUrl: env.MPESA_TIMEOUT_URL,

  // Transaction Types
  transactionType: "CustomerPayBillOnline", // For PayBill
  // transactionType: "CustomerBuyGoodsOnline", // For Till Number

  // Environment
  environment: env.MPESA_ENV,
} as const;

// API Endpoints
export const mpesaEndpoints = {
  // Authentication
  auth: "/oauth/v1/generate?grant_type=client_credentials",

  // STK Push
  stkPush: "/mpesa/stkpush/v1/processrequest",
  stkPushQuery: "/mpesa/stkpushquery/v1/query",

  // B2C
  b2c: "/mpesa/b2c/v1/paymentrequest",

  // Transaction Status
  transactionStatus: "/mpesa/transactionstatus/v1/query",

  // Account Balance
  accountBalance: "/mpesa/accountbalance/v1/query",

  // Reversal
  reversal: "/mpesa/reversal/v1/request",
} as const;

// Response Codes
export const mpesaResponseCodes = {
  SUCCESS: "0",
  INSUFFICIENT_FUNDS: "1",
  LESS_THAN_MINIMUM: "2",
  MORE_THAN_MAXIMUM: "3",
  WOULD_EXCEED_DAILY_LIMIT: "4",
  WOULD_EXCEED_MONTHLY_LIMIT: "5",
  INVALID_AMOUNT: "6",
  INVALID_ACCOUNT: "7",
  INVALID_PHONE: "8",
  INVALID_SHORTCODE: "9",
  INVALID_REFERENCE: "10",
  INVALID_TRANSACTION: "11",
  INVALID_ENVIRONMENT: "12",
  INVALID_CREDENTIALS: "13",
  INVALID_TIMESTAMP: "14",
  INVALID_PASSKEY: "15",
  INVALID_CHECKSUM: "16",
  INVALID_CALLBACK: "17",
  INVALID_TIMEOUT: "18",
  INVALID_TRANSACTION_TYPE: "19",
  INVALID_BUSINESS_SHORTCODE: "20",
} as const;

// Transaction Status
export const mpesaTransactionStatus = {
  SUCCESS: "Success",
  FAILED: "Failed",
  PENDING: "Pending",
  CANCELLED: "Cancelled",
  REVERSED: "Reversed",
  TIMEOUT: "Timeout",
} as const;
