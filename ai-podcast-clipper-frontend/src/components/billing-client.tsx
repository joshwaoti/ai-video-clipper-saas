"use client";

import type { VariantProps } from "class-variance-authority";
import { ArrowLeftIcon, CheckIcon, Loader2 } from "lucide-react"; // Added Loader2
import Link from "next/link";
import { useState } from "react"; // Added useState
import { initiateMpesaPayment, type CreditPack } from "~/actions/mpesa";
import { Button, buttonVariants } from "~/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "~/components/ui/card";
import { cn } from "~/lib/utils";
import { toast } from "sonner";
import { useRouter } from "next/navigation"; // Added for potential refresh later

interface PricingPlan {
  title: string;
  price: string;
  description: string;
  features: string[];
  buttonText: string;
  buttonVariant: VariantProps<typeof buttonVariants>["variant"];
  isPopular?: boolean;
  savePercentage?: string;
  creditPack: CreditPack;
}

const plans: PricingPlan[] = [
  {
    title: "Small Pack",
    price: "KES 999", // Assuming KES 1 was for testing and real price is 999
    description: "Perfect for occasional podcast creators",
    features: ["50 credits", "No expiration", "Download all clips"],
    buttonText: "Buy 50 credits",
    buttonVariant: "outline",
    creditPack: "small",
  },
  {
    title: "Medium Pack",
    price: "KES 2,499",
    description: "Best value for regular podcasters",
    features: ["150 credits", "No expiration", "Download all clips"],
    buttonText: "Buy 150 credits",
    buttonVariant: "default",
    isPopular: true,
    savePercentage: "Save 17%",
    creditPack: "medium",
  },
  {
    title: "Large Pack",
    price: "KES 6,999",
    description: "Ideal for podcast studios and agencies",
    features: ["500 credits", "No expiration", "Download all clips"],
    buttonText: "Buy 500 credits",
    buttonVariant: "outline",
    isPopular: false,
    savePercentage: "Save 30%",
    creditPack: "large",
  },
];

interface PricingCardProps {
  plan: PricingPlan;
  userPhoneNumber: string | null | undefined;
  onPaymentSuccessfullyInitiated?: (message: string) => void; // Added prop
}

function PricingCard({ plan, userPhoneNumber, onPaymentSuccessfullyInitiated }: PricingCardProps) {
  const [isProcessing, setIsProcessing] = useState(false); // Added loading state

  const handlePayment = async () => {
    if (!userPhoneNumber) {
      toast.error("Phone number is missing.", {
        description: "Please update your profile with your MPESA phone number.",
      });
      return;
    }
    setIsProcessing(true);
    try {
      const result = await initiateMpesaPayment(plan.creditPack);
      if (result.success) {
        // toast.success("STK Push Initiated", { // Removed this toast, parent will handle
        //   description: result.message || "Check your phone to complete payment.",
        // });
        if (onPaymentSuccessfullyInitiated && result.message) {
          onPaymentSuccessfullyInitiated(result.message);
        }
      } else {
        toast.error(result.error ?? "Payment initiation failed. Please try again.");
      }
    } catch (error) {
      toast.error("An unexpected error occurred while trying to initiate payment.");
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <Card
      className={cn(
        "relative flex flex-col",
        plan.isPopular && "border-primary border-2",
      )}
    >
      {plan.isPopular && (
        <div className="bg-primary text-primary-foreground absolute top-0 left-1/2 -translate-x-1/2 -translate-y-1/2 transform rounded-full px-3 py-1 text-sm font-medium whitespace-nowrap">
          Most Popular
        </div>
      )}
      <CardHeader className="flex-1">
        <CardTitle>{plan.title}</CardTitle>
        <div className="text-4xl font-bold">{plan.price} </div>
        {plan.savePercentage && (
          <p className="text-sm font-medium text-green-600">
            {plan.savePercentage}
          </p>
        )}
        <CardDescription>{plan.description}</CardDescription>
      </CardHeader>
      <CardContent className="space-y-2">
        <ul className="text-muted-foreground space-y-2 text-sm">
          {plan.features.map((feature, index) => (
            <li key={index} className="flex items-center gap-2">
              <CheckIcon className="text-primary size-4" />
              {feature}
            </li>
          ))}
        </ul>
      </CardContent>
      <CardFooter>
        <Button
          variant={plan.buttonVariant}
          className="w-full"
          onClick={handlePayment}
          disabled={isProcessing || !userPhoneNumber}
        >
          {isProcessing ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              Processing...
            </>
          ) : (
            plan.buttonText
          )}
        </Button>
      </CardFooter>
    </Card>
  );
}

interface BillingClientProps {
  userPhoneNumber: string | null | undefined;
  initialCredits: number;
}

export function BillingClient({ userPhoneNumber, initialCredits }: BillingClientProps) {
  const router = useRouter(); // Added router

  const handleSuccessfulInitiation = (initiationMessage: string) => {
    router.refresh();
    toast.success("STK Push Sent!", {
      description: `${initiationMessage} Please complete the payment on your phone. Your balance will update after confirmation.`,
      duration: 8000,
    });
  };

  return (
    <div className="mx-auto flex flex-col space-y-8 px-4 py-12">
      <div className="relative flex items-center justify-center gap-4">
        <Button
          className="absolute top-0 left-0"
          variant="outline"
          size="icon"
          asChild
        >
          <Link href="/dashboard">
            <ArrowLeftIcon className="size-4" />
          </Link>
        </Button>
        <div className="space-y-2 text-center">
          <h1 className="text-2xl font-bold tracking-tight sm:text-4xl">
            Buy Credits
          </h1>
          <p className="text-muted-foreground">
            Purchase credits to generate more podcast clips. The more credits
            you buy, the better the value.
          </p>
          <p className="text-lg font-semibold">Your current balance: {initialCredits} credits</p>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-8 md:grid-cols-3">
        {plans.map((plan) => (
          <PricingCard
            key={plan.title}
            plan={plan}
            userPhoneNumber={userPhoneNumber}
            onPaymentSuccessfullyInitiated={handleSuccessfulInitiation}
          />
        ))}
      </div>

      <div className="bg-muted/50 rounded-lg p-6">
        <h3 className="mb-4 text-lg font-semibold">How credits work</h3>
        <ul className="text-muted-foreground list-disc space-y-2 pl-5 text-sm">
          <li>1 credit = 1 minute of podcast processing</li>
          <li>
            The program will create around 1 clip per 5 minutes of podcast
          </li>
          <li>Credits never expire and can be used anytime</li>
          <li>Longer podcasts require more credits based on duration</li>
          <li>All packages are one-time purchases (not subscription)</li>
        </ul>
      </div>
    </div>
  );
}
