import { NextResponse } from "next/server";
import { db } from "~/server/db";
import { MpesaService } from "~/lib/mpesa/service";

export async function POST(req: Request) {
  try {
    const body = await req.json();
    const mpesaService = new MpesaService();

    // Verify the callback is from MPESA
    if (!body.Body?.stkCallback) {
      return new NextResponse("Invalid callback format", { status: 400 });
    }

    const { ResultCode, ResultDesc, CallbackMetadata } = body.Body.stkCallback;

    if (ResultCode === 0) {
      // Payment successful
      const { Amount, MpesaReceiptNumber, TransactionDate, PhoneNumber } =
        CallbackMetadata;

      // Find the pending transaction
      const transaction = await db.transaction.findFirst({
        where: {
          status: "pending",
          amount: Amount,
        },
        orderBy: {
          createdAt: "desc",
        },
      });

      if (transaction) {
        // Update transaction status
        await db.transaction.update({
          where: { id: transaction.id },
          data: {
            status: "completed",
          },
        });

        // Add credits to user
        await db.user.update({
          where: { id: transaction.userId },
          data: {
            credits: {
              increment: transaction.credits,
            },
          },
        });
      }
    } else {
      // Payment failed
      const transaction = await db.transaction.findFirst({
        where: {
          status: "pending",
        },
        orderBy: {
          createdAt: "desc",
        },
      });

      if (transaction) {
        await db.transaction.update({
          where: { id: transaction.id },
          data: {
            status: "failed",
          },
        });
      }
    }

    return new NextResponse(null, { status: 200 });
  } catch (error) {
    console.error("Error processing MPESA webhook:", error);
    return new NextResponse("Webhook error", { status: 500 });
  }
}
