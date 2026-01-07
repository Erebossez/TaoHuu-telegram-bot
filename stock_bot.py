import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters
import yfinance as yf
import matplotlib.pyplot as plt
from io import BytesIO
from datetime import datetime, timedelta
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# ดาวน์โหลด vader_lexicon อัตโนมัติ
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    print("กำลังดาวน์โหลด vader_lexicon ครั้งแรก... รอสักครู่")
    nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

import os
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

def calculate_rsi(series, period=40):
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1] if not rsi.empty else 'N/A'

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        'สวัสดีครับ! 📈\n'
        'พิมพ์ ticker เช่น NVDA เพื่อดูข้อมูลเดี่ยว\n'
        'หรือเปรียบเทียบสูงสุด 4 หุ้นด้วย / เช่น NVDA/PLTR หรือ NVDA/AAPL/TSLA/MSFT\n'
        'ข้อมูลครบ + กราฟเปรียบเทียบ + RSI 40 + sentiment ข่าว\n'
        'ตัวอย่าง: NVDA หรือ NVDA/AAPL/TSLA'
    )

async def get_stock(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip().upper()
    
    # นับจำนวน /
    if '/' in text:
        tickers = [t.strip() for t in text.split('/') if t.strip()]
        if 1 < len(tickers) <= 4:
            await compare_multiple_stocks(update, tickers)
            return
        elif len(tickers) > 4:
            await update.message.reply_text('รองรับเปรียบเทียบสูงสุด 4 หุ้นครับ 😅\nตัวอย่าง: NVDA/AAPL/TSLA/MSFT')
            return
    
    # โหมดเดี่ยว
    ticker = text
    await analyze_single_stock(update, ticker)

async def analyze_single_stock(update: Update, ticker: str):
    await update.message.reply_text(f'กำลังวิเคราะห์ {ticker}... รอสักครู่ ⏳')
    
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        name = info.get('longName', ticker)
        price = info.get('regularMarketPrice', 'N/A')
        change_pct = info.get('regularMarketChangePercent', 0)
        change_str = f"{change_pct:.2f}%" if isinstance(change_pct, float) else 'N/A'
        market_cap = info.get('marketCap')
        market_cap_str = f"${market_cap / 1e12:.2f}T" if market_cap and market_cap >= 1e12 else \
                         f"${market_cap / 1e9:.2f}B" if market_cap else 'N/A'
        
        pe = info.get('trailingPE', 'N/A')
        target = info.get('targetMeanPrice', 'N/A')
        
        msg = f"📊 {name} ({ticker})\n\n"
        msg += f"ราคา: ${price}   |   วันนี้: {change_str}\n"
        msg += f"มูลค่าตลาด: {market_cap_str}\n"
        msg += f"P/E: {pe}   |   เป้าหมาย Analyst: ${target}\n\n"
        
        # RSI 40
        hist = stock.history(period="1y")
        if not hist.empty:
            rsi40 = calculate_rsi(hist['Close'])
            msg += f"📌 RSI 40: {rsi40:.2f}\n"
            if rsi40 > 70:
                msg += "→ Overbought (อาจย่อตัว) ⚠️\n\n"
            elif rsi40 < 30:
                msg += "→ Oversold (โอกาสเด้ง) ✅\n\n"
            else:
                msg += "→ ปกติ ⚖️\n\n"
        
        # ข่าว + sentiment (ย่อ)
        sentiment_summary = await get_sentiment_summary(stock)
        msg += f"😊 Sentiment ข่าว: {sentiment_summary}\n"
        
        await update.message.reply_text(msg)
        
        # กราฟเดี่ยว
        if not hist.empty:
            hist_3mo = hist.tail(90)
            plt.figure(figsize=(12, 6))
            plt.plot(hist_3mo.index, hist_3mo['Close'], color='blue', linewidth=2)
            plt.title(f'{ticker} - 3-Month Price Chart', fontsize=16)
            plt.xlabel('Date')
            plt.ylabel('Price (USD)')
            plt.legend([ticker])
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            bio = BytesIO()
            plt.savefig(bio, format='png')
            bio.seek(0)
            await update.message.reply_photo(photo=bio, caption=f'📉 {ticker} 3-Month Chart')
            plt.close()
    
    except Exception as e:
        await update.message.reply_text(f'ผิดพลาด: {str(e)}')

async def compare_multiple_stocks(update: Update, tickers: list):
    num = len(tickers)
    await update.message.reply_text(f'กำลังเปรียบเทียบ {"/".join(tickers)} ({num} หุ้น)... รอสักครู่ ⏳')
    
    stocks = []
    infos = []
    histories = []
    colors = ['blue', 'orange', 'green', 'red']
    
    for t in tickers:
        try:
            s = yf.Ticker(t)
            i = s.info
            h = s.history(period="1y")
            stocks.append(s)
            infos.append(i)
            histories.append(h)
        except:
            await update.message.reply_text(f'ไม่พบข้อมูล {t}')
            return
    
    msg = f"📊 เปรียบเทียบ {' / '.join(tickers)}\n\n"
    
    # ตารางเปรียบเทียบ
    msg += "ราคา: " + " | ".join(f"${i.get('regularMarketPrice', 'N/A')}" for i in infos) + "\n"
    msg += "วันนี้: " + " | ".join(f"{i.get('regularMarketChangePercent', 0):.2f}%" for i in infos) + "\n"
    msg += "P/E: " + " | ".join(f"{i.get('trailingPE', 'N/A')}" for i in infos) + "\n"
    msg += "Market Cap: " + " | ".join(
        f"${i.get('marketCap', 0)/1e12:.2f}T" if i.get('marketCap', 0) >= 1e12 else f"${i.get('marketCap', 0)/1e9:.2f}B" if i.get('marketCap') else 'N/A'
        for i in infos
    ) + "\n\n"
    
    # RSI 40
    msg += "📌 RSI 40:\n"
    for t, h in zip(tickers, histories):
        if not h.empty:
            rsi = calculate_rsi(h['Close'])
            msg += f"{t}: {rsi:.2f}\n"
        else:
            msg += f"{t}: N/A\n"
    msg += "\n"
    
    # Sentiment รวม
    msg += "😊 Sentiment ข่าวรวม:\n"
    for t, s in zip(tickers, stocks):
        summary = await get_sentiment_summary(s)
        msg += f"{t}: {summary}\n"
    
    await update.message.reply_text(msg)
    
    # กราฟเปรียบเทียบ (สูงสุด 4 สี)
    plt.figure(figsize=(14, 7))
    for i, (t, h) in enumerate(zip(tickers, histories)):
        if not h.empty:
            plt.plot(h.index, h['Close'], label=t, color=colors[i], linewidth=2)
    
    plt.title(f"{' vs '.join(tickers)} - 1-Year Price Chart", fontsize=16)
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    bio = BytesIO()
    plt.savefig(bio, format='png')
    bio.seek(0)
    await update.message.reply_photo(photo=bio, caption=f'📉 เปรียบเทียบราคา {", ".join(tickers)} ย้อนหลัง 1 ปี')
    plt.close()

async def get_sentiment_summary(stock):
    try:
        news = stock.news[:5]
        scores = []
        for n in news:
            text = n.get('title', '') + " " + n.get('summary', '')
            scores.append(sia.polarity_scores(text)['compound'])
        avg = sum(scores) / len(scores) if scores else 0
        label = "Strong Positive ✅" if avg > 0.2 else "Mild Positive ✅" if avg > 0 else "Mild Negative ⚠️" if avg > -0.2 else "Strong Negative ⚠️"
        return f"{avg:.3f} → {label}"
    except:
        return "N/A"

if __name__ == '__main__':
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, get_stock))
    print("Bot พร้อมแล้ว! เปรียบเทียบสูงสุด 4 หุ้นด้วย / (เช่น NVDA/AAPL/TSLA/MSFT)")
    app.run_polling()
