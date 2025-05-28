import { mpesaConfig, mpesaEndpoints, mpesaResponseCodes } from "./config";
import {
  generateMpesaToken,
  generateMpesaPassword,
  generateMpesaTimestamp,
  formatPhoneNumber,
  generateTransactionReference,
  isValidMpesaPhone,
  handleMpesaError,
} from "./utils";
import type {
  MpesaSTKPushResponse,
  MpesaSTKPushQueryResponse,
  MpesaB2CResponse,
  MpesaTransactionStatusResponse,
  MpesaErrorResponse,
} from "./types";

/**
 * MPESA Service class for handling all MPESA operations
 */
export class MpesaService {
  private token: string | null = null;
  private tokenExpiry: Date | null = null;

  /**
   * Get a valid MPESA API token
   */
  private async getToken(): Promise<string> {
    if (this.token && this.tokenExpiry && this.tokenExpiry > new Date()) {
      return this.token;
    }

    this.token = await generateMpesaToken();
    // Token expires in 1 hour
    this.tokenExpiry = new Date(Date.now() + 3600000);
    return this.token;
  }

  /**
   * Initiate STK Push payment
   */
  async initiateSTKPush(
    phoneNumber: string,
    amount: number,
    accountReference: string,
    transactionDesc: string,
  ): Promise<MpesaSTKPushResponse> {
    try {
      const token = await this.getToken();
      const timestamp = generateMpesaTimestamp();
      const password = generateMpesaPassword();
      const checkoutRequestId = generateTransactionReference();

      const response = await fetch(
        `${mpesaConfig.baseUrl}${mpesaEndpoints.stkPush}`,
        {
          method: "POST",
          headers: {
            Authorization: `Bearer ${token}`,
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            BusinessShortCode: mpesaConfig.businessShortCode,
            Password: password,
            Timestamp: timestamp,
            TransactionType: mpesaConfig.transactionType,
            Amount: amount,
            PartyA: formatPhoneNumber(phoneNumber),
            PartyB: mpesaConfig.businessShortCode,
            PhoneNumber: formatPhoneNumber(phoneNumber),
            CallBackURL: mpesaConfig.callbackUrl,
            AccountReference: accountReference,
            TransactionDesc: transactionDesc,
            CheckoutRequestID: checkoutRequestId,
          }),
        },
      );

      if (!response.ok) {
        const errorData = (await response
          .json()
          .catch(() => null)) as MpesaErrorResponse | null;
        console.error("MPESA STK Push Error:", {
          status: response.status,
          statusText: response.statusText,
          errorData,
        });
        throw new Error(
          `Failed to initiate STK Push: ${response.statusText}${
            errorData ? ` - ${errorData.errorMessage}` : ""
          }`,
        );
      }

      const data = (await response.json()) as MpesaSTKPushResponse;
      return data;
    } catch (error) {
      console.error("MPESA STK Push Exception:", error);
      throw error;
    }
  }

  /**
   * Query STK Push status
   */
  async querySTKPushStatus(checkoutRequestId: string) {
    try {
      const token = await this.getToken();
      const timestamp = generateMpesaTimestamp();
      const password = generateMpesaPassword();

      const payload = {
        BusinessShortCode: mpesaConfig.businessShortCode,
        Password: password,
        Timestamp: timestamp,
        CheckoutRequestID: checkoutRequestId,
      };

      const response = await fetch(
        `${mpesaConfig.baseUrl}${mpesaEndpoints.stkPushQuery}`,
        {
          method: "POST",
          headers: {
            Authorization: `Bearer ${token}`,
            "Content-Type": "application/json",
          },
          body: JSON.stringify(payload),
        },
      );

      if (!response.ok) {
        throw new Error("Failed to query STK Push status");
      }

      const data = (await response.json()) as MpesaSTKPushQueryResponse;
      return {
        success: true,
        resultCode: data.ResultCode,
        resultDesc: data.ResultDesc,
        transactionId: data.TransactionId,
        amount: data.Amount,
        mpesaReceiptNumber: data.MpesaReceiptNumber,
        transactionDate: data.TransactionDate,
        phoneNumber: data.PhoneNumber,
      };
    } catch (error) {
      throw handleMpesaError(error);
    }
  }

  /**
   * Initiate B2C payment
   */
  async initiateB2CPayment(
    phoneNumber: string,
    amount: number,
    remarks: string,
    occasion = "",
  ) {
    try {
      if (!isValidMpesaPhone(phoneNumber)) {
        throw new Error("Invalid phone number format");
      }

      const token = await this.getToken();
      const transactionRef = generateTransactionReference();

      const payload = {
        InitiatorName: mpesaConfig.businessShortCode,
        SecurityCredential: generateMpesaPassword(),
        CommandID: "BusinessPayment",
        Amount: amount,
        PartyA: mpesaConfig.businessShortCode,
        PartyB: formatPhoneNumber(phoneNumber),
        Remarks: remarks,
        QueueTimeOutURL: mpesaConfig.timeoutUrl,
        ResultURL: mpesaConfig.callbackUrl,
        Occasion: occasion,
        TransactionID: transactionRef,
      };

      const response = await fetch(
        `${mpesaConfig.baseUrl}${mpesaEndpoints.b2c}`,
        {
          method: "POST",
          headers: {
            Authorization: `Bearer ${token}`,
            "Content-Type": "application/json",
          },
          body: JSON.stringify(payload),
        },
      );

      if (!response.ok) {
        throw new Error("Failed to initiate B2C payment");
      }

      const data = (await response.json()) as MpesaB2CResponse;
      return {
        success: true,
        conversationId: data.ConversationID,
        originatorConversationId: data.OriginatorConversationID,
        responseCode: data.ResponseCode,
        responseDescription: data.ResponseDescription,
      };
    } catch (error) {
      throw handleMpesaError(error);
    }
  }

  /**
   * Query transaction status
   */
  async queryTransactionStatus(
    transactionId: string,
    remarks = "Transaction status query",
  ) {
    try {
      const token = await this.getToken();
      const timestamp = generateMpesaTimestamp();
      const password = generateMpesaPassword();

      const payload = {
        Initiator: mpesaConfig.businessShortCode,
        SecurityCredential: password,
        CommandID: "TransactionStatusQuery",
        TransactionID: transactionId,
        PartyA: mpesaConfig.businessShortCode,
        IdentifierType: "4",
        ResultURL: mpesaConfig.callbackUrl,
        QueueTimeOutURL: mpesaConfig.timeoutUrl,
        Remarks: remarks,
        Occasion: "Transaction status query",
      };

      const response = await fetch(
        `${mpesaConfig.baseUrl}${mpesaEndpoints.transactionStatus}`,
        {
          method: "POST",
          headers: {
            Authorization: `Bearer ${token}`,
            "Content-Type": "application/json",
          },
          body: JSON.stringify(payload),
        },
      );

      if (!response.ok) {
        throw new Error("Failed to query transaction status");
      }

      const data = (await response.json()) as MpesaTransactionStatusResponse;
      return {
        success: true,
        conversationId: data.ConversationID,
        originatorConversationId: data.OriginatorConversationID,
        responseCode: data.ResponseCode,
        responseDescription: data.ResponseDescription,
      };
    } catch (error) {
      throw handleMpesaError(error);
    }
  }
}
