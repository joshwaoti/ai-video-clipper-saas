import { NextResponse } from "next/server";
import { db } from "~/server/db";
import { Prisma } from "@prisma/client";

interface CallbackMetadata {
  Amount: number;
  MpesaReceiptNumber: string;
  TransactionDate: string;
}

interface STKCallback {
  MerchantRequestID: string;
  CheckoutRequestID: string;
  ResultCode: number;
  ResultDesc: string;
  CallbackMetadata?: CallbackMetadata;
}

interface CallbackBody {
  Body: {
    stkCallback: STKCallback;
  };
}

export async function POST(request: Request) {
  try {
    // Log request details
    console.log("Callback received at:", new Date().toISOString());
    console.log(
      "Request headers:",
      Object.fromEntries(request.headers.entries()),
    );

    const rawBody = await request.text();
    console.log("Raw request body:", rawBody);

    const data = JSON.parse(rawBody) as CallbackBody;
    console.log("Parsed callback data:", JSON.stringify(data, null, 2));

    // Extract the relevant data from the callback
    const {
      Body: {
        stkCallback: {
          MerchantRequestID,
          CheckoutRequestID,
          ResultCode,
          ResultDesc,
          CallbackMetadata,
        },
      },
    } = data;

    console.log("Processing callback for:", {
      MerchantRequestID,
      CheckoutRequestID,
      ResultCode,
      ResultDesc,
    });

    // Find the transaction
    const transaction = await db.transaction.findFirst({
      where: {
        checkoutRequestId: CheckoutRequestID,
        merchantRequestId: MerchantRequestID,
      },
    });

    if (!transaction) {
      console.error("Transaction not found:", {
        CheckoutRequestID,
        MerchantRequestID,
      });
      return NextResponse.json(
        { error: "Transaction not found" },
        { status: 404 },
      );
    }

    console.log("Found transaction:", {
      id: transaction.id,
      userId: transaction.userId,
      credits: transaction.credits,
      status: transaction.status,
    });

    // If payment was successful
    if (ResultCode === 0 && CallbackMetadata) {
      const { Amount, MpesaReceiptNumber, TransactionDate } = CallbackMetadata;
      console.log("Payment successful:", {
        Amount,
        MpesaReceiptNumber,
        TransactionDate,
      });

      // Update transaction status
      await db.transaction.update({
        where: { id: transaction.id },
        data: {
          status: "completed",
        },
      });

      // Add credits to user's account
      const updatedUser = await db.user.update({
        where: { id: transaction.userId },
        data: {
          credits: {
            increment: transaction.credits,
          },
        },
      });

      console.log("Updated user credits:", {
        userId: updatedUser.id,
        newCredits: updatedUser.credits,
      });

      return NextResponse.json({ success: true });
    } else {
      console.log("Payment failed:", {
        ResultCode,
        ResultDesc,
      });

      // Payment failed
      await db.transaction.update({
        where: { id: transaction.id },
        data: {
          status: "failed",
        },
      });

      return NextResponse.json({ success: false, error: ResultDesc });
    }
  } catch (error) {
    console.error("MPESA Callback Error:", error);
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 },
    );
  }
}
