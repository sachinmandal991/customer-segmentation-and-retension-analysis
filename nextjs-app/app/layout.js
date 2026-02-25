import './globals.css'

export const metadata = {
  title: 'Customer Analytics Dashboard',
  description: 'AI-Powered Customer Segmentation & Churn Prediction',
}

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
