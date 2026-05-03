import json
import logging
from config import parse_args
from nexus.nexus_trainer.trainer import NexusTrainer

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

if __name__ == "__main__":
    cfg = parse_args()
    log.info("Starting training with config:\n" + json.dumps(cfg.__dict__, indent=2, default=str))
    
    trainer = NexusTrainer(cfg)
    trainer.train()