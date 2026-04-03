import json 
from pathlib import Path
from time import perf_counter

META_DATA_PATH = './arxiv_dataset/arxiv-metadata-oai-snapshot.json'
SAVE_DIR = './arxiv_dataset/math_by_category_dedup'

class ArXivMathDataPipeline:
    def __init__(self, metadata_path=META_DATA_PATH, save_dir=SAVE_DIR):
        self.metadata_path = Path(metadata_path)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)

    def filter_math_papers(self, log_interval=50_000):
        """
        Extracts math subset of arXiv metadata.
        """
        papers = []
        # all math metadata save location
        save_jsonl = self.save_dir / 'all_math_papers_metadata.jsonl'

        math_papers = 0
        total_papers = 0
        start = t0 = perf_counter()
        with open(self.metadata_path, 'r') as f, open(save_jsonl, 'a') as out_f:
            while (line := f.readline().strip()):
                total_papers += 1
                record = json.loads(line)
                categories = record['categories']
                if 'math.' not in categories:
                    continue
                math_papers += 1
                papers.append(line+'\n')
                if len(papers) % 1000 == 0:
                    out_f.write(''.join(papers))
                    papers = []
                    if len(papers) % log_interval == 0:

                        # stats
                        percentage = round(100 * math_papers/total_papers, 2)
                        dt = perf_counter() - t0

                        print(f'Found {math_papers} math papers | {percentage}% total papers | this shard took {dt} seconds')
                        t0 = perf_counter()
            out_f.write(''.join(papers))
            dt_min = round((perf_counter()-start)/60, 1)
            print(f'End. Found {math_papers} math papers. Took {dt_min} min')

    def filter_math_by_category(self, math_jsonl_path=None):
        if math_jsonl_path is None:
            math_jsonl_path = self.save_dir / 'all_math_papers_metadata.jsonl'
        

    def algebra_filters(self):
        return {
            'math.AC': 'Commutative Algebra',
            'math.CT': 'Category Theory',
            'math.KT': 'K-Theory & Homology',
            'math.OA': 'Operator Algebras',
            'math.QA': 'Quantum Algebra',
            'math.GR': 'Group Theory',
            'math.RA': 'Rings & Algebras',
            'math.RT': 'Representation Theory',
            'math.NT': 'Number Theory',
        }
    
    def geometry_filters(self):
        return {
            'math.AG': 'Algebraic Geometry',
            'math.DG': 'Differential Geometry',
            'math.MG': 'Metric Geometry',
            'math.SG': 'Symplectic Geometry'
        }
    
    def topology_filters(self):
        return {
            'math.AT': 'Algebraic Topology',
            'math.GN': 'General Topology',
            'math.GT': 'Geometric Topology',
        }

    def analysis_filters(self):
        return {
            'math.AP': 'Analysis of PDEs',
            'math.CA': 'Classical Analysis & ODEs',
            'math.CV': 'Complex Variables',
            'math.DS': 'Dynamical Systems',
            'math.FA': 'Functional Analysis',        
            'math.SP': 'Spectral Theory',
        }
    
    def discrete_math_filters(self):
        return {
            'math.CO': 'Combinatorics',
            'math.LO': 'Logic',
        }
    
if __name__ == '__main__':
    arxiv = ArXivMathDataPipeline()
    arxiv.filter_math_papers()
