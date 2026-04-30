"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                   KANNADA OCR DIAGNOSTIC MODULE                              ║
║                                                                              ║
║  Purpose: Identify sources of Telugu characters, fragmentation, and         ║
║           extraction issues in the Kannada OCR system                       ║
║                                                                              ║
║  Author: Diagnostic Tool for MTech Thesis Project                           ║
║  Date: 2026-01-05                                                           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import os
import json
import unicodedata
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import traceback

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))


# ═══════════════════════════════════════════════════════════════════════════
#                           CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

class DiagnosticConfig:
    """Configuration for diagnostic tests"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        
        # Output directories
        self.diagnostics_dir = self.base_dir / "diagnostics"
        self.reports_dir = self.diagnostics_dir / "reports"
        self.raw_outputs_dir = self.diagnostics_dir / "raw_outputs"
        self.unicode_analysis_dir = self.diagnostics_dir / "unicode_analysis"
        self.logs_dir = self.diagnostics_dir / "logs"
        
        # Model and data paths
        self.model_path = self.base_dir / "models" / "hybrid_kannada_ocr_20251204_151641.pth"
        self.class_mapping_path = self.base_dir / "models" / "class_mapping_20251204_151641.json"
        self.credentials_path = self.base_dir / "credentials" / "kannadaocrextraction-c6b23b356a5d.json"
        self.test_pdf = self.base_dir / "samples" / "sample6.pdf"
        
        # Create output directories
        self._create_directories()
        
        # Timestamp for this run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def _create_directories(self):
        """Create all required output directories"""
        for directory in [self.diagnostics_dir, self.reports_dir, 
                         self.raw_outputs_dir, self.unicode_analysis_dir, 
                         self.logs_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            
    def get_log_file(self) -> Path:
        """Get path for current log file"""
        return self.logs_dir / f"diagnostic_{self.timestamp}.log"
        
    def get_report_file(self, format: str = "html") -> Path:
        """Get path for report file"""
        return self.reports_dir / f"diagnostic_report_{self.timestamp}.{format}"


# ═══════════════════════════════════════════════════════════════════════════
#                        UNICODE ANALYSIS TOOLS
# ═══════════════════════════════════════════════════════════════════════════

class UnicodeAnalyzer:
    """Analyze Unicode properties of characters"""
    
    # Unicode ranges for Indic scripts
    KANNADA_RANGE = (0x0C80, 0x0CFF)
    TELUGU_RANGE = (0x0C00, 0x0C7F)
    TAMIL_RANGE = (0x0B80, 0x0BFF)
    MALAYALAM_RANGE = (0x0D00, 0x0D7F)
    
    # Kannada specific categories
    KANNADA_VOWELS = list(range(0x0C85, 0x0C8C + 1)) + list(range(0x0C8E, 0x0C90 + 1)) + list(range(0x0C92, 0x0C94 + 1))
    KANNADA_CONSONANTS = list(range(0x0C95, 0x0CB9 + 1))
    KANNADA_VOWEL_SIGNS = list(range(0x0CBE, 0x0CC4 + 1)) + list(range(0x0CC6, 0x0CC8 + 1)) + list(range(0x0CCA, 0x0CCD + 1))
    
    @staticmethod
    def analyze_character(char: str) -> Dict:
        """
        Comprehensive analysis of a single character
        
        Returns:
            Dict with Unicode properties, script detection, etc.
        """
        if not char:
            return {"error": "Empty character"}
            
        codepoint = ord(char)
        
        try:
            name = unicodedata.name(char, "UNKNOWN")
        except ValueError:
            name = "UNNAMED"
            
        category = unicodedata.category(char)
        
        analysis = {
            "character": char,
            "unicode": f"U+{codepoint:04X}",
            "decimal": codepoint,
            "name": name,
            "category": category,
            "category_description": UnicodeAnalyzer._get_category_description(category),
            "script": UnicodeAnalyzer.detect_script(char),
            "is_combining": unicodedata.combining(char) > 0,
            "is_vowel_sign": UnicodeAnalyzer.is_vowel_sign(char),
            "is_standalone": UnicodeAnalyzer.is_standalone_char(char),
            "visual_warning": UnicodeAnalyzer._get_visual_warning(char, codepoint)
        }
        
        return analysis
    
    @staticmethod
    def _get_category_description(category: str) -> str:
        """Get human-readable description of Unicode category"""
        categories = {
            "Lo": "Letter, Other (Base character)",
            "Mn": "Mark, Nonspacing (Combining character)",
            "Mc": "Mark, Spacing Combining (Vowel sign)",
            "Nd": "Number, Decimal Digit",
            "Po": "Punctuation, Other",
            "Zs": "Separator, Space",
            "Cc": "Control character"
        }
        return categories.get(category, f"Unknown ({category})")
    
    @staticmethod
    def detect_script(char: str) -> str:
        """Detect which Indic script a character belongs to"""
        codepoint = ord(char)
        
        if UnicodeAnalyzer.KANNADA_RANGE[0] <= codepoint <= UnicodeAnalyzer.KANNADA_RANGE[1]:
            return "KANNADA"
        elif UnicodeAnalyzer.TELUGU_RANGE[0] <= codepoint <= UnicodeAnalyzer.TELUGU_RANGE[1]:
            return "TELUGU"
        elif UnicodeAnalyzer.TAMIL_RANGE[0] <= codepoint <= UnicodeAnalyzer.TAMIL_RANGE[1]:
            return "TAMIL"
        elif UnicodeAnalyzer.MALAYALAM_RANGE[0] <= codepoint <= UnicodeAnalyzer.MALAYALAM_RANGE[1]:
            return "MALAYALAM"
        elif codepoint < 128:
            return "ASCII"
        else:
            return "OTHER"
    
    @staticmethod
    def is_vowel_sign(char: str) -> bool:
        """Check if character is a Kannada vowel sign (combining mark)"""
        return ord(char) in UnicodeAnalyzer.KANNADA_VOWEL_SIGNS
    
    @staticmethod
    def is_standalone_char(char: str) -> bool:
        """Check if character can stand alone (not a combining mark)"""
        category = unicodedata.category(char)
        return category not in ['Mn', 'Mc', 'Me']
    
    @staticmethod
    def _get_visual_warning(char: str, codepoint: int) -> Optional[str]:
        """Get warning if character has visual issues"""
        category = unicodedata.category(char)
        
        if category in ['Mn', 'Mc'] and UnicodeAnalyzer.detect_script(char) == "KANNADA":
            return "⚠️ VOWEL SIGN - Should not appear alone!"
        elif UnicodeAnalyzer.detect_script(char) == "TELUGU":
            return "❌ TELUGU CHARACTER - Should be Kannada!"
        elif codepoint == 0x0CCD:  # Halant
            return "⚠️ HALANT - Forms consonant clusters"
        
        return None
    
    @staticmethod
    def analyze_text(text: str) -> Dict:
        """Analyze entire text for script distribution and issues"""
        if not text:
            return {"error": "Empty text"}
        
        char_analyses = [UnicodeAnalyzer.analyze_character(c) for c in text]
        
        # Count scripts
        script_counts = defaultdict(int)
        for analysis in char_analyses:
            script_counts[analysis['script']] += 1
        
        # Count issues
        issues = {
            "isolated_vowel_signs": 0,
            "telugu_characters": 0,
            "unknown_characters": 0,
            "fragmentation_indicators": []
        }
        
        for i, analysis in enumerate(char_analyses):
            if analysis.get('visual_warning'):
                if "VOWEL SIGN" in analysis['visual_warning']:
                    issues['isolated_vowel_signs'] += 1
                    issues['fragmentation_indicators'].append({
                        "position": i,
                        "character": analysis['character'],
                        "issue": "Isolated vowel sign"
                    })
                elif "TELUGU" in analysis['visual_warning']:
                    issues['telugu_characters'] += 1
                    issues['fragmentation_indicators'].append({
                        "position": i,
                        "character": analysis['character'],
                        "issue": "Telugu character"
                    })
        
        return {
            "total_characters": len(text),
            "script_distribution": dict(script_counts),
            "character_analyses": char_analyses,
            "issues": issues,
            "kannada_percentage": (script_counts['KANNADA'] / len(text) * 100) if text else 0,
            "telugu_percentage": (script_counts['TELUGU'] / len(text) * 100) if text else 0
        }


# ═══════════════════════════════════════════════════════════════════════════
#                       OCR COMPONENT DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════════

class OCRDiagnostics:
    """Test individual OCR components and hybrid system"""
    
    def __init__(self, config: DiagnosticConfig):
        self.config = config
        self.analyzer = UnicodeAnalyzer()
        self.log_file = config.get_log_file()
        
        # Initialize log
        self._log("=" * 80)
        self._log("KANNADA OCR DIAGNOSTICS")
        self._log(f"Started: {datetime.now()}")
        self._log("=" * 80)
    
    def _log(self, message: str):
        """Write to log file and print to console"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_message + '\n')
    
    def test_custom_model_only(self, pdf_path: Optional[str] = None) -> Dict:
        """
        Test extraction using ONLY the custom DenseNet-DeiT model
        
        This isolates the custom model to see if it produces Telugu or fragmentation
        """
        self._log("\n" + "=" * 80)
        self._log("TEST 1: CUSTOM MODEL ONLY")
        self._log("=" * 80)
        
        if pdf_path is None:
            pdf_path = str(self.config.test_pdf)
        
        try:
            # Import extraction module
            from extraction import HybridExtractor
            
            # Create extractor with GCV disabled
            self._log("Creating extractor with Google Cloud Vision DISABLED...")
            
            # Note: You'll need to modify HybridExtractor to accept a flag
            # For now, we'll document what the output should look like
            
            self._log("⚠️  NOTE: To fully isolate custom model, you need to:")
            self._log("   1. Modify extraction.py HybridExtractor class")
            self._log("   2. Add parameter: use_gcv=False")
            self._log("   3. Skip GCV API calls when use_gcv=False")
            
            # Try to extract anyway and capture output
            extractor = HybridExtractor(
                model_path=str(self.config.model_path),
                class_mapping_path=str(self.config.class_mapping_path),
                credentials_path=str(self.config.credentials_path)
            )
            
            self._log(f"Processing PDF: {pdf_path}")
            
            # Extract
            result = extractor.extract(pdf_path)
            extracted_text = result.get('text', '') if isinstance(result, dict) else str(result)
            
            self._log(f"\nExtracted Text ({len(extracted_text)} characters):")
            self._log("-" * 80)
            self._log(extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text)
            self._log("-" * 80)
            
            # Analyze Unicode
            self._log("\nAnalyzing Unicode properties...")
            analysis = self.analyzer.analyze_text(extracted_text)
            
            self._log(f"\nScript Distribution:")
            for script, count in analysis['script_distribution'].items():
                percentage = (count / analysis['total_characters'] * 100)
                self._log(f"  {script}: {count} ({percentage:.1f}%)")
            
            self._log(f"\nIssues Detected:")
            self._log(f"  Isolated vowel signs: {analysis['issues']['isolated_vowel_signs']}")
            self._log(f"  Telugu characters: {analysis['issues']['telugu_characters']}")
            
            # Save raw output
            output_file = self.config.raw_outputs_dir / "custom_model_output.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(extracted_text)
            self._log(f"\nRaw output saved to: {output_file}")
            
            # Save analysis
            analysis_file = self.config.unicode_analysis_dir / "custom_model_analysis.json"
            with open(analysis_file, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, indent=2, ensure_ascii=False)
            self._log(f"Unicode analysis saved to: {analysis_file}")
            
            return {
                "success": True,
                "text": extracted_text,
                "analysis": analysis,
                "output_file": str(output_file)
            }
            
        except Exception as e:
            self._log(f"\n❌ ERROR in custom model test: {str(e)}")
            self._log(traceback.format_exc())
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def test_gcv_only(self, pdf_path: Optional[str] = None) -> Dict:
        """
        Test extraction using ONLY Google Cloud Vision API
        
        This identifies if GCV is introducing Telugu characters
        """
        self._log("\n" + "=" * 80)
        self._log("TEST 2: GOOGLE CLOUD VISION ONLY")
        self._log("=" * 80)
        
        if pdf_path is None:
            pdf_path = str(self.config.test_pdf)
        
        try:
            from google.cloud import vision
            from PIL import Image
            import io
            
            # Initialize GCV client
            self._log("Initializing Google Cloud Vision client...")
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = str(self.config.credentials_path)
            client = vision.ImageAnnotatorClient()
            
            # Convert PDF to image (first page)
            self._log(f"Processing PDF: {pdf_path}")
            from pdf2image import pdfimages
            
            # Note: You may need to install pdf2image
            # For now, we'll document the process
            
            self._log("⚠️  NOTE: To extract with GCV only:")
            self._log("   1. Convert PDF to image")
            self._log("   2. Call GCV with language_hints=['kn']")
            self._log("   3. Check if output contains Telugu")
            
            # Placeholder for actual implementation
            self._log("\n📝 Implementation needed:")
            self._log("   See manual GCV test instructions below")
            
            return {
                "success": False,
                "message": "Implementation placeholder - see logs for manual test instructions"
            }
            
        except Exception as e:
            self._log(f"\n❌ ERROR in GCV test: {str(e)}")
            self._log(traceback.format_exc())
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def test_hybrid(self, pdf_path: Optional[str] = None) -> Dict:
        """
        Test full hybrid extraction (your current system)
        
        This shows the actual production output
        """
        self._log("\n" + "=" * 80)
        self._log("TEST 3: HYBRID EXTRACTION (PRODUCTION)")
        self._log("=" * 80)
        
        if pdf_path is None:
            pdf_path = str(self.config.test_pdf)
        
        try:
            # Use your existing extraction pipeline
            from extraction_with_correction import extract_and_correct_pdf
            
            self._log(f"Running full extraction pipeline on: {pdf_path}")
            
            result = extract_and_correct_pdf(
                pdf_path=pdf_path,
                model_path=str(self.config.model_path),
                class_mapping_path=str(self.config.class_mapping_path),
                credentials_path=str(self.config.credentials_path)
            )
            
            extracted_text = result.get('corrected_text', result.get('text', ''))
            
            self._log(f"\nExtracted Text ({len(extracted_text)} characters):")
            self._log("-" * 80)
            self._log(extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text)
            self._log("-" * 80)
            
            # Analyze
            analysis = self.analyzer.analyze_text(extracted_text)
            
            self._log(f"\nScript Distribution:")
            for script, count in analysis['script_distribution'].items():
                percentage = (count / analysis['total_characters'] * 100)
                self._log(f"  {script}: {count} ({percentage:.1f}%)")
            
            self._log(f"\nIssues Detected:")
            self._log(f"  Isolated vowel signs: {analysis['issues']['isolated_vowel_signs']}")
            self._log(f"  Telugu characters: {analysis['issues']['telugu_characters']}")
            
            # Save outputs
            output_file = self.config.raw_outputs_dir / "hybrid_output.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(extracted_text)
            self._log(f"\nRaw output saved to: {output_file}")
            
            analysis_file = self.config.unicode_analysis_dir / "hybrid_analysis.json"
            with open(analysis_file, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, indent=2, ensure_ascii=False)
            self._log(f"Unicode analysis saved to: {analysis_file}")
            
            return {
                "success": True,
                "text": extracted_text,
                "analysis": analysis,
                "output_file": str(output_file),
                "full_result": result
            }
            
        except Exception as e:
            self._log(f"\n❌ ERROR in hybrid test: {str(e)}")
            self._log(traceback.format_exc())
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def compare_outputs(self, custom_result: Dict, gcv_result: Dict, hybrid_result: Dict) -> Dict:
        """Compare outputs from all three extraction methods"""
        self._log("\n" + "=" * 80)
        self._log("COMPARATIVE ANALYSIS")
        self._log("=" * 80)
        
        comparison = {
            "custom_vs_hybrid": {},
            "gcv_vs_hybrid": {},
            "quality_assessment": {}
        }
        
        # Compare script purity
        if custom_result.get('success') and custom_result.get('analysis'):
            custom_kannada = custom_result['analysis']['kannada_percentage']
            hybrid_kannada = hybrid_result['analysis']['kannada_percentage']
            
            self._log(f"\nKannada Purity:")
            self._log(f"  Custom Model: {custom_kannada:.1f}%")
            self._log(f"  Hybrid System: {hybrid_kannada:.1f}%")
            
            if hybrid_kannada < custom_kannada:
                self._log("  ⚠️  Hybrid system has LOWER Kannada purity than custom model!")
                self._log("  → GCV or merge logic may be introducing Telugu")
        
        # Compare fragmentation
        custom_frag = custom_result.get('analysis', {}).get('issues', {}).get('isolated_vowel_signs', 0)
        hybrid_frag = hybrid_result.get('analysis', {}).get('issues', {}).get('isolated_vowel_signs', 0)
        
        self._log(f"\nFragmentation (Isolated Vowel Signs):")
        self._log(f"  Custom Model: {custom_frag}")
        self._log(f"  Hybrid System: {hybrid_frag}")
        
        if hybrid_frag > custom_frag:
            self._log("  ⚠️  Hybrid system has MORE fragmentation!")
            self._log("  → Merge logic may be breaking aksharas")
        
        return comparison


# ═══════════════════════════════════════════════════════════════════════════
#                         REPORT GENERATION
# ═══════════════════════════════════════════════════════════════════════════

class DiagnosticReporter:
    """Generate comprehensive diagnostic reports"""
    
    def __init__(self, config: DiagnosticConfig):
        self.config = config
    
    def generate_html_report(self, results: Dict) -> str:
        """Generate beautiful HTML report"""
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Kannada OCR Diagnostic Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            text-align: center;
        }}
        .section {{
            background: white;
            padding: 25px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .success {{ color: #10b981; font-weight: bold; }}
        .warning {{ color: #f59e0b; font-weight: bold; }}
        .error {{ color: #ef4444; font-weight: bold; }}
        .code {{
            background: #1e1e1e;
            color: #d4d4d4;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
            font-family: 'Consolas', 'Monaco', monospace;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #e5e7eb;
        }}
        th {{
            background: #f3f4f6;
            font-weight: 600;
        }}
        .metric {{
            display: inline-block;
            padding: 10px 20px;
            margin: 5px;
            border-radius: 5px;
            font-weight: bold;
        }}
        .metric.good {{ background: #d1fae5; color: #065f46; }}
        .metric.warning {{ background: #fef3c7; color: #92400e; }}
        .metric.bad {{ background: #fee2e2; color: #991b1b; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔬 Kannada OCR Diagnostic Report</h1>
        <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    </div>
"""
        
        # Add test results
        if 'hybrid_test' in results and results['hybrid_test'].get('success'):
            analysis = results['hybrid_test']['analysis']
            
            html += f"""
    <div class="section">
        <h2>📊 Quick Summary</h2>
        <div style="text-align: center;">
            <div class="metric {'good' if analysis['kannada_percentage'] > 95 else 'warning' if analysis['kannada_percentage'] > 80 else 'bad'}">
                Kannada: {analysis['kannada_percentage']:.1f}%
            </div>
            <div class="metric {'good' if analysis['telugu_percentage'] == 0 else 'bad'}">
                Telugu: {analysis['telugu_percentage']:.1f}%
            </div>
            <div class="metric {'good' if analysis['issues']['isolated_vowel_signs'] == 0 else 'warning'}">
                Fragmentation: {analysis['issues']['isolated_vowel_signs']} issues
            </div>
        </div>
    </div>
    
    <div class="section">
        <h2>🔍 Detailed Analysis</h2>
        <h3>Script Distribution</h3>
        <table>
            <tr>
                <th>Script</th>
                <th>Count</th>
                <th>Percentage</th>
            </tr>
"""
            
            for script, count in analysis['script_distribution'].items():
                percentage = (count / analysis['total_characters'] * 100)
                html += f"""
            <tr>
                <td>{script}</td>
                <td>{count}</td>
                <td>{percentage:.2f}%</td>
            </tr>
"""
            
            html += """
        </table>
        
        <h3>Issues Detected</h3>
        <ul>
"""
            
            if analysis['issues']['isolated_vowel_signs'] > 0:
                html += f"<li class='warning'>Found {analysis['issues']['isolated_vowel_signs']} isolated vowel signs (fragmentation)</li>"
            else:
                html += "<li class='success'>No isolated vowel signs detected ✓</li>"
            
            if analysis['issues']['telugu_characters'] > 0:
                html += f"<li class='error'>Found {analysis['issues']['telugu_characters']} Telugu characters!</li>"
            else:
                html += "<li class='success'>No Telugu characters detected ✓</li>"
            
            html += """
        </ul>
    </div>
"""
        
        # Sample text
        if 'hybrid_test' in results and results['hybrid_test'].get('text'):
            text = results['hybrid_test']['text'][:500]
            html += f"""
    <div class="section">
        <h2>📄 Sample Extracted Text</h2>
        <div class="code">
{text}
        </div>
    </div>
"""
        
        # Recommendations
        html += """
    <div class="section">
        <h2>💡 Recommendations</h2>
        <ol>
"""
        
        if 'hybrid_test' in results and results['hybrid_test'].get('success'):
            analysis = results['hybrid_test']['analysis']
            
            if analysis['telugu_percentage'] > 0:
                html += """
            <li class="error">
                <strong>Telugu characters detected!</strong>
                <ul>
                    <li>Check Google Cloud Vision language hints</li>
                    <li>Add <code>language_hints=['kn']</code> to GCV API call</li>
                    <li>Test GCV isolation to confirm source</li>
                </ul>
            </li>
"""
            
            if analysis['issues']['isolated_vowel_signs'] > 5:
                html += """
            <li class="warning">
                <strong>Significant fragmentation detected!</strong>
                <ul>
                    <li>Implement character combiner post-processing</li>
                    <li>Check if custom model outputs individual codepoints</li>
                    <li>Review hybrid merge logic</li>
                </ul>
            </li>
"""
            
            if analysis['kannada_percentage'] > 95 and analysis['issues']['isolated_vowel_signs'] < 5:
                html += """
            <li class="success">
                <strong>Overall quality is good!</strong>
                <ul>
                    <li>High Kannada purity</li>
                    <li>Minimal fragmentation</li>
                    <li>Focus on UI improvements</li>
                </ul>
            </li>
"""
        
        html += """
        </ol>
    </div>
    
    <div class="section">
        <h2>📁 Generated Files</h2>
        <ul>
"""
        
        html += f"<li>Diagnostic log: <code>{self.config.get_log_file()}</code></li>"
        html += f"<li>Raw outputs: <code>{self.config.raw_outputs_dir}</code></li>"
        html += f"<li>Unicode analysis: <code>{self.config.unicode_analysis_dir}</code></li>"
        
        html += """
        </ul>
    </div>
</body>
</html>
"""
        
        # Save HTML report
        report_file = self.config.get_report_file('html')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"\n✅ HTML report generated: {report_file}")
        return str(report_file)
    
    def generate_json_report(self, results: Dict) -> str:
        """Generate JSON report for programmatic access"""
        report_file = self.config.get_report_file('json')
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"✅ JSON report generated: {report_file}")
        return str(report_file)


# ═══════════════════════════════════════════════════════════════════════════
#                           MAIN RUNNER
# ═══════════════════════════════════════════════════════════════════════════

def run_full_diagnostics(pdf_path: Optional[str] = None):
    """
    Run complete diagnostic suite
    
    Args:
        pdf_path: Optional path to PDF file. If None, uses default sample6.pdf
    
    Returns:
        Dict with all test results
    """
    print("\n" + "="*80)
    print("🔬 KANNADA OCR DIAGNOSTIC SUITE")
    print("="*80)
    
    # Initialize
    config = DiagnosticConfig()
    diagnostics = OCRDiagnostics(config)
    reporter = DiagnosticReporter(config)
    
    print(f"\n📁 Output directory: {config.diagnostics_dir}")
    print(f"📝 Log file: {config.get_log_file()}")
    
    if pdf_path is None:
        pdf_path = str(config.test_pdf)
        print(f"📄 Test PDF: {pdf_path}")
    
    # Verify PDF exists
    if not Path(pdf_path).exists():
        print(f"\n❌ ERROR: PDF not found at {pdf_path}")
        print("Please provide a valid PDF path or place sample6.pdf in samples/ folder")
        return None
    
    results = {}
    
    # Test 1: Custom model only
    print("\n" + "-"*80)
    print("🧪 Running Test 1: Custom Model Only")
    print("-"*80)
    custom_result = diagnostics.test_custom_model_only(pdf_path)
    results['custom_test'] = custom_result
    
    # Test 2: GCV only (placeholder for now)
    print("\n" + "-"*80)
    print("🧪 Running Test 2: Google Cloud Vision Only")
    print("-"*80)
    gcv_result = diagnostics.test_gcv_only(pdf_path)
    results['gcv_test'] = gcv_result
    
    # Test 3: Hybrid (current system)
    print("\n" + "-"*80)
    print("🧪 Running Test 3: Hybrid System (Production)")
    print("-"*80)
    hybrid_result = diagnostics.test_hybrid(pdf_path)
    results['hybrid_test'] = hybrid_result
    
    # Comparative analysis
    if custom_result.get('success') and hybrid_result.get('success'):
        comparison = diagnostics.compare_outputs(custom_result, gcv_result, hybrid_result)
        results['comparison'] = comparison
    
    # Generate reports
    print("\n" + "="*80)
    print("📊 GENERATING REPORTS")
    print("="*80)
    
    html_report = reporter.generate_html_report(results)
    json_report = reporter.generate_json_report(results)
    
    # Print summary
    print("\n" + "="*80)
    print("✅ DIAGNOSTICS COMPLETE")
    print("="*80)
    
    if hybrid_result.get('success'):
        analysis = hybrid_result['analysis']
        print(f"\n📊 Key Findings:")
        print(f"  • Kannada purity: {analysis['kannada_percentage']:.1f}%")
        print(f"  • Telugu contamination: {analysis['telugu_percentage']:.1f}%")
        print(f"  • Fragmentation issues: {analysis['issues']['isolated_vowel_signs']}")
        
        if analysis['telugu_percentage'] > 0:
            print(f"\n⚠️  WARNING: Telugu characters detected!")
            print(f"   → Check Google Cloud Vision language settings")
        
        if analysis['issues']['isolated_vowel_signs'] > 5:
            print(f"\n⚠️  WARNING: Significant fragmentation detected!")
            print(f"   → Implement character combiner post-processing")
        
        if analysis['kannada_percentage'] > 95 and analysis['issues']['isolated_vowel_signs'] < 5:
            print(f"\n✅ Overall quality is GOOD! Minor issues only.")
    
    print(f"\n📁 Reports saved to:")
    print(f"   • HTML: {html_report}")
    print(f"   • JSON: {json_report}")
    print(f"   • Log: {config.get_log_file()}")
    
    print("\n💡 Next steps:")
    print("   1. Open the HTML report in your browser")
    print("   2. Review Unicode analysis files")
    print("   3. Check recommendations section")
    
    return results


# ═══════════════════════════════════════════════════════════════════════════
#                         COMMAND LINE INTERFACE
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Kannada OCR Diagnostic Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default sample PDF
  python diagnostic_ocr.py
  
  # Run with custom PDF
  python diagnostic_ocr.py --pdf "path/to/your/file.pdf"
  
  # Analyze specific text
  python diagnostic_ocr.py --text "ಕನ್ನಡ ಪಠ್ಯ"
        """
    )
    
    parser.add_argument('--pdf', type=str, help='Path to PDF file to test')
    parser.add_argument('--text', type=str, help='Analyze specific text string')
    
    args = parser.parse_args()
    
    if args.text:
        # Quick text analysis mode
        print("\n🔍 Quick Text Analysis")
        print("="*80)
        analyzer = UnicodeAnalyzer()
        analysis = analyzer.analyze_text(args.text)
        
        print(f"\nText: {args.text}")
        print(f"Total characters: {analysis['total_characters']}")
        print(f"\nScript Distribution:")
        for script, count in analysis['script_distribution'].items():
            percentage = (count / analysis['total_characters'] * 100)
            print(f"  {script}: {count} ({percentage:.1f}%)")
        
        print(f"\nIssues:")
        print(f"  Isolated vowel signs: {analysis['issues']['isolated_vowel_signs']}")
        print(f"  Telugu characters: {analysis['issues']['telugu_characters']}")
        
        print(f"\nCharacter-by-character:")
        for char_analysis in analysis['character_analyses'][:20]:  # First 20 chars
            warning = char_analysis.get('visual_warning', '')
            print(f"  {char_analysis['character']} | {char_analysis['unicode']} | {char_analysis['script']} {warning}")
        
    else:
        # Full diagnostic suite
        run_full_diagnostics(args.pdf)