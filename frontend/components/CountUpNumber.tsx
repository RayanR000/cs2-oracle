'use client';

import { useEffect, useState } from 'react';
import { motion } from 'framer-motion';

interface CountUpNumberProps {
  from: number;
  to: number;
  decimals?: number;
  duration?: number;
  formatFn?: (value: number) => string;
}

export default function CountUpNumber({
  from,
  to,
  decimals = 0,
  duration = 1,
  formatFn,
}: CountUpNumberProps) {
  const [displayValue, setDisplayValue] = useState(from);

  useEffect(() => {
    let animationFrame: number;
    let startTime: number | null = null;

    const animate = (currentTime: number) => {
      if (startTime === null) {
        startTime = currentTime;
      }

      const elapsed = currentTime - startTime;
      const progress = Math.min(elapsed / (duration * 1000), 1);

      // Ease-out effect
      const easeProgress = 1 - Math.pow(1 - progress, 3);
      const current = from + (to - from) * easeProgress;

      setDisplayValue(current);

      if (progress < 1) {
        animationFrame = requestAnimationFrame(animate);
      }
    };

    animationFrame = requestAnimationFrame(animate);

    return () => cancelAnimationFrame(animationFrame);
  }, [from, to, duration]);

  const formattedValue = formatFn
    ? formatFn(displayValue)
    : displayValue.toFixed(decimals);

  return <span>{formattedValue}</span>;
}
