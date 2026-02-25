# Customer Analytics - Next.js Application

## 🚀 Quick Start

### 1. Install Dependencies
```bash
cd nextjs-app
npm install
```

### 2. Start Flask API (Required)
In another terminal:
```bash
cd ..
python api.py
```

### 3. Run Next.js Dev Server
```bash
npm run dev
```

Open: **http://localhost:3000**

## 📦 Tech Stack

### Frontend
- **Next.js 14** - React framework with App Router
- **React 18** - UI library  
- **Chart.js 4.4** - Data visualization
- **React-ChartJS-2** - React wrapper for Chart.js
- **Axios** - HTTP client

### Backend
- **Flask API** - Running on port 5000
- **Python ML Models** - KMeans, Logistic Regression, XGBoost

## 🎯 Features

✅ Server-Side Rendering (SSR)
✅ Client-Side Rendering (CSR)
✅ React Hooks (useState, useEffect)
✅ Async/Await API calls
✅ Real-time data fetching
✅ Interactive charts
✅ Responsive design
✅ Component-based architecture

## 📁 Project Structure

```
nextjs-app/
├── app/
│   ├── layout.js       # Root layout
│   ├── page.js         # Main dashboard
│   └── globals.css     # Global styles
├── package.json        # Dependencies
└── next.config.js      # Next.js config
```

## 🔧 Available Scripts

```bash
npm run dev      # Development server (port 3000)
npm run build    # Production build
npm start        # Production server
npm run lint     # ESLint
```

## 🌐 API Endpoints Used

- GET `/api/stats` - Dashboard statistics
- POST `/api/predict` - Churn prediction
- POST `/api/segment` - Segment prediction
- GET `/api/customer/:id` - Customer details
- GET `/api/segments` - Segment analysis

## 📝 Sample Customer IDs

- `7590-VHVEG` (Low risk)
- `3668-QPYBK` (High risk)
- `9237-HQITU` (Medium risk)

## 🚀 Production Deployment

### Build
```bash
npm run build
```

### Deploy to Vercel
```bash
npm install -g vercel
vercel deploy
```

### Deploy to Netlify
```bash
npm run build
# Upload .next folder to Netlify
```

## 🔗 Environment Variables

Create `.env.local`:
```
NEXT_PUBLIC_API_URL=http://localhost:5000
```

## 💡 Why Next.js?

- ⚡ Fast page loads with SSR
- 🔍 SEO optimized
- 📦 Automatic code splitting
- 🔥 Hot module replacement
- 🎨 CSS-in-JS support
- 📱 Mobile responsive
- 🚀 Production ready

## 🆚 Next.js vs Plain HTML

| Feature | Next.js | Plain HTML |
|---------|---------|------------|
| Framework | React | Vanilla JS |
| Rendering | SSR + CSR | CSR only |
| Routing | Built-in | Manual |
| SEO | Excellent | Limited |
| Performance | Optimized | Manual |
| Build Tools | Included | None |
| Components | Reusable | Manual |

## 📚 Learn More

- [Next.js Documentation](https://nextjs.org/docs)
- [React Documentation](https://react.dev)
- [Chart.js Documentation](https://www.chartjs.org)

## ✅ Success!

Your Next.js app is now running with modern React architecture! 🎉
