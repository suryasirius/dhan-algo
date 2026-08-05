import logging
import config

logger = logging.getLogger("LivePaperTrading.Broker")

class PaperBroker:
    def __init__(self, capital: float = config.INITIAL_CAPITAL):
        self.capital = capital
        self.positions = []
        self.trade_history = []
        logger.info(f"Paper Broker initialized with capital: ?{self.capital:,.2f} [PAPER MODE ONLY]")

    def execute_order(self, symbol: str, strike: float, option_type: str, strategy: str, entry_price: float, target: float, stop_loss: float):
        if not config.PAPER_TRADING_ONLY:
            raise RuntimeError("CRITICAL ERROR: Real trading blocked by system architecture.")
        
        exec_price = entry_price + config.SLIPPAGE_POINTS
        logger.info(f"[PAPER FILL] {strategy} | {symbol} {strike} {option_type} @ ?{exec_price:.2f}")
        
        position = {
            "symbol": symbol,
            "strike": strike,
            "option_type": option_type,
            "strategy": strategy,
            "entry_price": exec_price,
            "target": target,
            "stop_loss": stop_loss,
            "status": "OPEN"
        }
        self.positions.append(position)
        return position

    def close_position(self, position: dict, exit_price: float, result: str, reason: str):
        exit_price -= config.SLIPPAGE_POINTS
        pnl_pts = (exit_price - position["entry_price"]) if position["option_type"] == "CE" else (position["entry_price"] - exit_price)
        pnl_inr = (pnl_pts * config.LOT_SIZE) - (config.BROKERAGE_PER_ORDER * 2)
        
        self.capital += pnl_inr
        trade_record = {
            **position,
            "exit_price": exit_price,
            "pnl_points": round(pnl_pts, 2),
            "pnl_inr": round(pnl_inr, 2),
            "result": result,
            "reason": reason,
            "closing_capital": round(self.capital, 2)
        }
        self.trade_history.append(trade_record)
        logger.info(f"[PAPER EXIT] {position['strategy']} | Result: {result} | PnL: ?{pnl_inr:,.2f} | Capital: ?{self.capital:,.2f}")
        return trade_record
