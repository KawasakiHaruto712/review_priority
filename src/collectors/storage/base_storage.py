"""
基底ストレージモジュール

データ保存の基底クラスを提供します。
"""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
from abc import ABC, abstractmethod


class BaseStorage(ABC):
    """ストレージ基底クラス"""
    
    def __init__(self, output_dir: Path):
        """
        Args:
            output_dir: 出力ディレクトリ
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    @abstractmethod
    def save_component_data(self, component: str, data: List[Dict[str, Any]]):
        """
        コンポーネントデータを保存
        
        Args:
            component: コンポーネント名
            data: 保存するデータ
        """
        pass
    
    def save_json(self, path: Path, data: Any):
        """
        JSONファイルを保存
        
        Args:
            path: 保存先パス
            data: 保存するデータ
        """
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def save_csv(self, path: Path, data: List[Dict[str, Any]]):
        """
        CSVファイルを保存
        
        Args:
            path: 保存先パス
            data: 保存するデータ
        """
        if data:
            df = pd.DataFrame(data)
            df.to_csv(path, index=False)
