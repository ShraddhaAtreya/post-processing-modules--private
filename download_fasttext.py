import urllib.request
import gzip
import shutil
import os

def download_indicnlp_kannada():
    """Download Kannada FastText from IndicNLP (AI4Bharat)"""
    
    # Alternative source: IndicNLP Project
    # These are trained specifically on Indian language corpora
    url = "https://storage.googleapis.com/ai4bharat-public-indic-nlp-corpora/embedding/fasttext/indicnlp.ft.kn.300.vec.gz"
    
    output_dir = "models"
    compressed = os.path.join(output_dir, "indicnlp.ft.kn.300.vec.gz")
    final = os.path.join(output_dir, "fasttext_kannada.vec")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Downloading IndicNLP FastText Kannada...")
    print(f"Source: AI4Bharat / IndicNLP")
    print(f"URL: {url}")
    print("Size: ~500 MB")
    
    def progress(count, block_size, total_size):
        if total_size > 0:
            percent = int(count * block_size * 100 / total_size)
            print(f"\r  Progress: {percent}%", end='')
        else:
            downloaded_mb = (count * block_size) / (1024**2)
            print(f"\r  Downloaded: {downloaded_mb:.1f} MB", end='')
    
    try:
        urllib.request.urlretrieve(url, compressed, progress)
        print("\n✅ Download complete!")
        
        print("Decompressing...")
        with gzip.open(compressed, 'rb') as f_in:
            with open(final, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        os.remove(compressed)
        
        # Verify
        print("Verifying file contains Kannada...")
        with open(final, 'r', encoding='utf-8') as f:
            header = f.readline().strip()
            print(f"  Header: {header}")
            
            # Check first 5 words
            kannada_found = 0
            for i in range(5):
                line = f.readline().strip()
                if line:
                    word = line.split()[0]
                    has_kannada = any('\u0C80' <= c <= '\u0CFF' for c in word)
                    if has_kannada:
                        kannada_found += 1
                        print(f"  ✓ {word}")
            
            if kannada_found >= 3:
                print(f"\n✅ SUCCESS! File contains Kannada")
                print(f"   File ready: {final}")
                return final
            else:
                print(f"\n❌ ERROR: File doesn't contain enough Kannada")
                return None
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    download_indicnlp_kannada()