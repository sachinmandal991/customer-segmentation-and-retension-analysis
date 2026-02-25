'use client'

import { useState, useEffect } from 'react'
import axios from 'axios'
import { Chart as ChartJS, ArcElement, CategoryScale, LinearScale, BarElement, PointElement, LineElement, Title, Tooltip, Legend } from 'chart.js'
import { Pie, Bar, Scatter } from 'react-chartjs-2'

ChartJS.register(ArcElement, CategoryScale, LinearScale, BarElement, PointElement, LineElement, Title, Tooltip, Legend)

const API_URL = 'http://localhost:5000'

export default function Home() {
  const [stats, setStats] = useState(null)
  const [segments, setSegments] = useState([])
  const [churnResult, setChurnResult] = useState(null)
  const [segmentResult, setSegmentResult] = useState(null)
  const [customerResult, setCustomerResult] = useState(null)

  useEffect(() => {
    loadStats()
  }, [])

  const loadStats = async () => {
    try {
      const res = await axios.get(`${API_URL}/api/stats`)
      setStats(res.data)
    } catch (error) {
      console.error('Error loading stats:', error)
    }
  }

  const loadSegments = async () => {
    try {
      const res = await axios.get(`${API_URL}/api/segments`)
      setSegments(res.data.segments)
    } catch (error) {
      console.error('Error loading segments:', error)
    }
  }

  const predictChurn = async (e) => {
    e.preventDefault()
    const customerId = e.target.customerId.value
    try {
      const res = await axios.post(`${API_URL}/api/predict`, { customer_id: customerId })
      setChurnResult(res.data)
    } catch (error) {
      setChurnResult({ error: error.response?.data?.error || 'Failed to connect' })
    }
  }

  const predictSegment = async (e) => {
    e.preventDefault()
    const customerId = e.target.customerId.value
    try {
      const res = await axios.post(`${API_URL}/api/segment`, { customer_id: customerId })
      setSegmentResult(res.data)
    } catch (error) {
      setSegmentResult({ error: error.response?.data?.error || 'Failed to connect' })
    }
  }

  const getCustomer = async (e) => {
    e.preventDefault()
    const customerId = e.target.customerId.value
    try {
      const res = await axios.get(`${API_URL}/api/customer/${customerId}`)
      setCustomerResult(res.data)
    } catch (error) {
      setCustomerResult({ error: error.response?.data?.error || 'Failed to connect' })
    }
  }

  return (
    <>
      <nav>
        <div className="nav-container">
          <div style={{ fontSize: '1.5em', fontWeight: 'bold', color: '#667eea' }}>
            📊 Customer Analytics
          </div>
          <ul className="nav-links">
            <li><a href="#dashboard">Dashboard</a></li>
            <li><a href="#predictions">Predictions</a></li>
            <li><a href="#segments">Segments</a></li>
            <li><a href="#visualize">Visualize</a></li>
            <li><a href="#about">About</a></li>
          </ul>
        </div>
      </nav>

      <div className="container">
        <div className="card" style={{ textAlign: 'center', marginTop: '20px' }} id="dashboard">
          <h1>🎯 Customer Analytics Dashboard</h1>
          <p style={{ color: '#666', fontSize: '1.1em' }}>AI-Powered Customer Segmentation & Churn Prediction</p>
        </div>

        {stats && (
          <div className="grid grid-4">
            <div className="stat-card">
              <h3 style={{ fontSize: '0.9em', color: '#666', marginBottom: '10px' }}>TOTAL CUSTOMERS</h3>
              <div className="stat-value">{stats.total_customers.toLocaleString()}</div>
            </div>
            <div className="stat-card">
              <h3 style={{ fontSize: '0.9em', color: '#666', marginBottom: '10px' }}>CHURN RATE</h3>
              <div className="stat-value">{stats.churn_rate.toFixed(1)}%</div>
            </div>
            <div className="stat-card">
              <h3 style={{ fontSize: '0.9em', color: '#666', marginBottom: '10px' }}>TOTAL REVENUE</h3>
              <div className="stat-value">${(stats.total_revenue / 1000000).toFixed(1)}M</div>
            </div>
            <div className="stat-card">
              <h3 style={{ fontSize: '0.9em', color: '#666', marginBottom: '10px' }}>HIGH RISK</h3>
              <div className="stat-value">{stats.high_risk_customers.toLocaleString()}</div>
            </div>
          </div>
        )}

        <div className="grid grid-2" id="predictions" style={{ marginTop: '30px' }}>
          <div className="card">
            <h2>🔮 Churn Prediction</h2>
            <form onSubmit={predictChurn}>
              <input name="customerId" placeholder="Customer ID (e.g., 7590-VHVEG)" required />
              <button type="submit" className="btn">Predict Churn</button>
            </form>
            {churnResult && (
              <div style={{ marginTop: '20px', padding: '15px', background: churnResult.error ? '#f8d7da' : '#d4edda', borderRadius: '8px' }}>
                {churnResult.error ? (
                  <p style={{ color: '#721c24' }}>{churnResult.error}</p>
                ) : (
                  <>
                    <p><strong>Customer:</strong> {churnResult.customer_id}</p>
                    <p><strong>Churn Probability:</strong> {(churnResult.churn_probability * 100).toFixed(1)}%</p>
                    <p><strong>Risk Level:</strong> <span style={{ color: churnResult.risk_level === 'High' ? 'red' : churnResult.risk_level === 'Medium' ? 'orange' : 'green' }}>{churnResult.risk_level}</span></p>
                    <p><strong>Segment:</strong> {churnResult.segment}</p>
                  </>
                )}
              </div>
            )}
          </div>

          <div className="card">
            <h2>👥 Segment Prediction</h2>
            <form onSubmit={predictSegment}>
              <input name="customerId" placeholder="Customer ID (e.g., 7590-VHVEG)" required />
              <button type="submit" className="btn">Predict Segment</button>
            </form>
            {segmentResult && (
              <div style={{ marginTop: '20px', padding: '15px', background: segmentResult.error ? '#f8d7da' : '#d4edda', borderRadius: '8px' }}>
                {segmentResult.error ? (
                  <p style={{ color: '#721c24' }}>{segmentResult.error}</p>
                ) : (
                  <>
                    <p><strong>Customer:</strong> {segmentResult.customer_id}</p>
                    <p><strong>Segment:</strong> {segmentResult.segment}</p>
                    <p><strong>RFM Score:</strong> {segmentResult.rfm_score.toFixed(1)}</p>
                    <p><strong>CLV:</strong> ${segmentResult.clv.toFixed(2)}</p>
                  </>
                )}
              </div>
            )}
          </div>

          <div className="card">
            <h2>🔍 Customer Details</h2>
            <form onSubmit={getCustomer}>
              <input name="customerId" placeholder="Customer ID (e.g., 7590-VHVEG)" required />
              <button type="submit" className="btn">Get Details</button>
            </form>
            {customerResult && (
              <div style={{ marginTop: '20px', padding: '15px', background: customerResult.error ? '#f8d7da' : '#d4edda', borderRadius: '8px' }}>
                {customerResult.error ? (
                  <p style={{ color: '#721c24' }}>{customerResult.error}</p>
                ) : (
                  <>
                    <p><strong>Customer:</strong> {customerResult.customer_id}</p>
                    <p><strong>Segment:</strong> {customerResult.segment}</p>
                    <p><strong>Risk:</strong> {customerResult.risk_level}</p>
                    <p><strong>CLV:</strong> ${customerResult.clv.toFixed(2)}</p>
                    <p><strong>Tenure:</strong> {customerResult.tenure.toFixed(0)} months</p>
                  </>
                )}
              </div>
            )}
          </div>

          <div className="card" id="segments">
            <h2>📊 Segment Analysis</h2>
            <button onClick={loadSegments} className="btn">Load Segments</button>
            {segments.length > 0 && (
              <table style={{ width: '100%', marginTop: '20px', borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ background: '#f8f9fa' }}>
                    <th style={{ padding: '12px', textAlign: 'left' }}>Segment</th>
                    <th style={{ padding: '12px', textAlign: 'left' }}>Customers</th>
                    <th style={{ padding: '12px', textAlign: 'left' }}>Churn %</th>
                  </tr>
                </thead>
                <tbody>
                  {segments.map((seg, i) => (
                    <tr key={i}>
                      <td style={{ padding: '12px' }}>{seg.segment_name}</td>
                      <td style={{ padding: '12px' }}>{seg.customer_count}</td>
                      <td style={{ padding: '12px' }}>{seg.churn_rate.toFixed(1)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </div>

        <div className="card" id="about" style={{ marginTop: '30px' }}>
          <h2>ℹ️ About</h2>
          <p style={{ lineHeight: '1.8', color: '#666' }}>
            AI-powered customer analytics platform built with <strong>Next.js 14</strong>, <strong>React 18</strong>, and <strong>Chart.js</strong>. 
            Connects to Flask API for real-time predictions using machine learning models (KMeans, Logistic Regression, XGBoost).
          </p>
        </div>
      </div>

      <footer>
        <div style={{ textAlign: 'center', color: '#666' }}>
          <p>&copy; 2024 Customer Analytics Platform | Built with Next.js</p>
        </div>
      </footer>
    </>
  )
}
