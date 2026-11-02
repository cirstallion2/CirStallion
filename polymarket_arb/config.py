import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class KrakenConfig:
    symbol: str = "XBTUSD"
    kline_interval: str = "5m"
    kline_limit: int = 200
    reconnect_delay_s: float = 1.0


@dataclass
class PolymarketConfig:
    clob_url: str = "https://clob.polymarket.com"
    private_key: str = field(default_factory=lambda: os.getenv("POLY_PRIVATE_KEY", ""))
    api_key: str = field(default_factory=lambda: os.getenv("POLY_API_KEY", ""))
    api_secret: str = field(default_factory=lambda: os.getenv("POLY_API_SECRET", ""))
    api_passphrase: str = field(default_factory=lambda: os.getenv("POLY_API_PASSPHRASE", ""))
    funder_address: str = field(default_factory=lambda: os.getenv("POLY_FUNDER_ADDRESS", ""))
    chain_id: int = 137  # Polygon mainnet


@dataclass
class CryptoQuantConfig:
    api_key: str = field(default_factory=lambda: os.getenv("CRYPTOQUANT_API_KEY", ""))
    base_url: str = "https://api.cryptoquant.com/v1"
    refresh_interval_s: float = 300.0  # Update every 5 minutes (hourly data)


@dataclass
class TradingViewConfig:
    exchange: str = "KRAKEN"
    screener: str = "crypto"
    symbol: str = "BTCUSD"
    interval: str = "5m"
    refresh_interval_s: float = 60.0


@dataclass
class RiskConfig:
    per_trade_risk_pct: float = 0.005   # 0.5% of portfolio per trade
    daily_cap_pct: float = 0.02          # 2% daily profit cap
    hard_stop_pct: float = 0.004         # 0.4% drawdown hard stop
    min_edge_pct: float = 0.003          # Minimum 0.3% CLOB lag to trade
    max_position_usdc: float = 500.0     # Hard cap per position in USDC
    min_position_usdc: float = 10.0      # Minimum order size (Polymarket minimum)
    portfolio_size_usdc: float = 10000.0  # Total portfolio size for risk %


@dataclass
class GraphConfig:
    num_nodes: int = 100
    num_edges: int = 180
    convergence_threshold: float = 0.65  # Min confidence to confirm trade direction
    force_iterations: int = 50
    spring_k: float = 0.12
    repulsion_k: float = 5.5
    damping: float = 0.85
    noise_sigma: float = 0.04            # Node value perturbation for realism


@dataclass
class BotConfig:
    kraken: KrakenConfig = field(default_factory=KrakenConfig)
    polymarket: PolymarketConfig = field(default_factory=PolymarketConfig)
    cryptoquant: CryptoQuantConfig = field(default_factory=CryptoQuantConfig)
    tradingview: TradingViewConfig = field(default_factory=TradingViewConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)

    # Execution
    target_execution_ms: int = 100        # <100ms execution target
    loop_interval_s: float = 0.02         # 20ms main loop (~50Hz)
    lag_window_s: float = 0.5             # Window to confirm CLOB lag persists
    min_liquidity_usdc: float = 500.0     # Minimum CLOB depth before trading

    # Market discovery
    btc_keyword: str = "BTC"
    target_expiry_minutes: int = 5

    # Dry-run: set to False only with valid credentials + funded wallet
    dry_run: bool = field(default_factory=lambda: os.getenv("DRY_RUN", "true").lower() != "false")

    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    log_file: str = "arb_bot.log"


CONFIG = BotConfig()
