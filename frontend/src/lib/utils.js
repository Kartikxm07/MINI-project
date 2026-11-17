import { clsx } from 'clsx'
import { twMerge } from 'tailwind-merge'

// Standard shadcn `cn` helper: merge class names and resolve Tailwind conflicts.
export function cn(...inputs) {
  return twMerge(clsx(inputs))
}
