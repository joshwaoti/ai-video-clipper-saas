export interface MpesaSTKPushResponse {
  MerchantRequestID: string;
  CheckoutRequestID: string;
  ResponseCode: string;
  ResponseDescription: string;
  CustomerMessage: string;
}

export interface MpesaSTKPushQueryResponse {
  ResultCode: string;
  ResultDesc: string;
  TransactionId: string;
  Amount: number;
  MpesaReceiptNumber: string;
  TransactionDate: string;
  PhoneNumber: string;
}

export interface MpesaB2CResponse {
  ConversationID: string;
  OriginatorConversationID: string;
  ResponseCode: string;
  ResponseDescription: string;
}

export interface MpesaTransactionStatusResponse {
  ConversationID: string;
  OriginatorConversationID: string;
  ResponseCode: string;
  ResponseDescription: string;
}

export interface MpesaErrorResponse {
  requestId: string;
  errorCode: string;
  errorMessage: string;
}
