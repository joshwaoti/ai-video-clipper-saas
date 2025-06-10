"use client";

import { useState, useTransition } from "react";
import { Button } from "~/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "~/components/ui/card";
import { RadioGroup, RadioGroupItem } from "~/components/ui/radio-group";
import { Label } from "~/components/ui/label";
import { toast } from "sonner";
import { initiateMpesaPayment, type CreditPack } from "~/actions/mpesa";

const CREDIT_PACK_OPTIONS: {
  value: CreditPack;
  label: string;
  price: string;
  credits: number;
}[] = [
  { value: "small", label: "Small Pack", price: "KES 1", credits: 50 }, // NOTE: Price KES 1 for testing, actual is KES 999 in actions/mpesa.ts
  { value: "medium", label: "Medium Pack", price: "KES 2,499", credits: 150 },
  { value: "large", label: "Large Pack", price: "KES 6,999", credits: 500 },
];

interface MpesaCheckoutFormProps {
  userPhoneNumber: string | null | undefined;
  onPaymentInitiated?: (customerMessage: string) => void;
  // These are for future enhancements where parent component handles polling/websockets
  // onPaymentSuccess?: () => void;
  // onPaymentError?: (error: string) => void;
}

export function MpesaCheckoutForm({
  userPhoneNumber,
  onPaymentInitiated,
}: MpesaCheckoutFormProps) {
  const [selectedPack, setSelectedPack] = useState<CreditPack>("small");
  const [isPending, startTransition] = useTransition();

  const handlePayment = () => {
    if (!userPhoneNumber) {
      toast.error("Error", {
        description: "Your phone number is not set. Please update your profile.",
      });
      return;
    }

    startTransition(async () => {
      try {
        const result = await initiateMpesaPayment(selectedPack);
        if (result.success) {
          toast.success("Payment Initiated", {
            description: result.message || "Check your phone to complete the payment.",
          });
          if (onPaymentInitiated && result.message) {
            onPaymentInitiated(result.message);
          }
        } else {
          toast.error("Payment Initiation Failed", {
            description: result.error || "An unknown error occurred.",
          });
        }
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : "An unexpected error occurred.";
        toast.error("Payment Error", {
          description: errorMessage,
        });
      }
    });
  };

  const selectedOptionDetails = CREDIT_PACK_OPTIONS.find(
    (opt) => opt.value === selectedPack,
  );

  return (
    <Card className="w-full max-w-md">
      <CardHeader>
        <CardTitle>Buy Credits with MPESA</CardTitle>
        <CardDescription>
          Select a credit pack and complete the payment on your phone.
          {userPhoneNumber ? ` We will send the STK push to ${userPhoneNumber}.` : " Your phone number is missing or not provided."}
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        <RadioGroup
          value={selectedPack}
          onValueChange={(value: string) => setSelectedPack(value as CreditPack)}
          className="space-y-2"
        >
          {CREDIT_PACK_OPTIONS.map((pack) => (
            <Label
              key={pack.value}
              htmlFor={pack.value} // Ensure this matches RadioGroupItem id
              className="flex cursor-pointer items-center justify-between rounded-md border-2 border-muted bg-popover p-4 hover:bg-accent hover:text-accent-foreground [&:has([data-state=checked])]:border-primary"
            >
              <div className="flex items-center space-x-3">
                <RadioGroupItem value={pack.value} id={pack.value} />
                <span>{pack.label}</span>
              </div>
              <div className="text-right">
                <div className="font-semibold">{pack.price}</div>
                <div className="text-sm text-muted-foreground">
                  {pack.credits} Credits
                </div>
              </div>
            </Label>
          ))}
        </RadioGroup>

        {selectedOptionDetails && (
          <div className="rounded-lg border bg-background p-4">
            <h3 className="mb-2 text-lg font-semibold">Summary</h3>
            <div className="flex justify-between">
              <span>Selected Pack:</span>
              <span>{selectedOptionDetails.label}</span>
            </div>
            <div className="flex justify-between">
              <span>Price:</span>
              <span>{selectedOptionDetails.price}</span>
            </div>
            <div className="flex justify-between font-bold">
              <span>Credits to Receive:</span>
              <span>{selectedOptionDetails.credits}</span>
            </div>
          </div>
        )}

        {!userPhoneNumber && (
          <p className="text-sm text-red-600">
            Please update your profile with a valid MPESA phone number to proceed.
          </p>
        )}
      </CardContent>
      <CardFooter>
        <Button
          onClick={handlePayment}
          disabled={isPending || !userPhoneNumber}
          className="w-full"
        >
          {isPending ? "Processing..." : `Pay ${selectedOptionDetails?.price || ""}`}
        </Button>
      </CardFooter>
    </Card>
  );
}

// Add this to ensure the toast hook is available
// You might need to install it if not already part of shadcn/ui setup:
// npm install sonner or check your existing toast setup
// For shadcn/ui, ensure you have Toaster component in your layout.tsx
// and the use-toast hook is correctly imported from components/ui/use-toast.
// For this example, `sonner` is used directly.
