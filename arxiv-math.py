import json
from contextlib import ExitStack
from pathlib import Path
from time import perf_counter
from collections import defaultdict

import bisect
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from typing import Iterator


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
        with open(self.metadata_path, 'r') as f, open(save_jsonl, 'w') as out_f:
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
        
        # pure math categories
        categories = self.pure_math_categories()

        category_tags = set(categories.keys())
        histogram = {k: 0 for k in categories.keys()}
        
        pure_math_dir = self.save_dir / 'pure_math'
        pure_math_dir.mkdir(parents=True, exist_ok=True)

        num_excluded_records = 0
        with ExitStack() as stack, open(math_jsonl_path, 'r') as f_input:
            file_handles = {
                k: stack.enter_context(open(pure_math_dir/f'{v.replace(' ', '_')}.jsonl', 'w', buffering=512))
                for k, v in categories.items()
            }
            start_t = t0 = perf_counter()
            i = 0
            for line in f_input:
                i += 1
                record = json.loads(line.strip())
                
                # only match one category to avoid duplication
                record_categories = category_tags & set(record['categories'].split())
                if record_categories:
                    cat = record_categories.pop()
                    file_handles[cat].write(line)
                    histogram[cat] += 1
                else:
                    num_excluded_records += 1
                
                if i % 100_000 == 0:
                    dt = round(perf_counter()-t0)
                    print(f'Processed {i} math records | latest 100_000 records took {dt} seconds.')
                    t0 = perf_counter()

            dt = round(perf_counter()-start_t)

            # stats 
            percent_excluded = round(100*num_excluded_records/i, 1)
            print(f'Complete in {dt} seconds. Saved {i-num_excluded_records} records. Excluded {num_excluded_records} ({percent_excluded}%) records.')
        
        with open(self.save_dir / 'pure_math_category_stats.json', 'w') as stats_file:
            stats_file.write(json.dumps(histogram))

        # saved dir
        return str(pure_math_dir)
    
    # def plot_pure_math_histogram(self, json_path=None, save_path=None):
    #     if json_path is None:
    #         json_path = self.save_dir / 'pure_math_category_stats.json'
        
    #     if save_path is None:
    #         save_path = str(self.save_dir / 'pure_math_category_histogram.pdf')

    #     with open(json_path, 'r') as f:
    #         histogram = json.load(f)

    #     categories = self.pure_math_categories()
    #     cats = [categories[k] for k in histogram.keys()]
    #     counts = list(histogram.values())

    #     sorted_pairs = sorted(zip(cats, counts), key=lambda x: x[1])
    #     cats, counts = zip(*sorted_pairs)

    #     plt.figure(figsize=(10, 8))
    #     plt.barh(cats, counts)
    #     plt.xlabel('Number of Papers')
    #     plt.ylabel('Category')
    #     plt.title('Pure Math Papers by Category')
    #     plt.tight_layout()
    #     plt.savefig(save_path)

    #     print(f'Saved histogram at {save_path}')
    #     return save_path


    def plot_pure_math_histogram(self, json_path=None, save_path=None):
        if json_path is None:
            json_path = self.save_dir / 'pure_math_category_stats.json'
        
        if save_path is None:
            save_path = str(self.save_dir / 'pure_math_category_histogram.pdf')

        with open(json_path, 'r') as f:
            histogram = json.load(f)

        categories = self.pure_math_categories()
        cats = [categories[k] for k in histogram.keys()]
        codes = list(histogram.keys())
        counts = list(histogram.values())

        # sort ascending
        sorted_data = sorted(zip(cats, codes, counts), key=lambda x: x[2])
        cats, codes, counts = zip(*sorted_data)

        fig, ax = plt.subplots(figsize=(11, 8))
        fig.patch.set_facecolor('#fafafa')
        ax.set_facecolor('#fafafa')

        # color gradient
        norm = plt.Normalize(min(counts), max(counts))
        colors = [plt.cm.GnBu(norm(c) * 0.7 + 0.3) for c in counts]

        bars = ax.barh(range(len(cats)), counts, color=colors,
                    edgecolor='white', linewidth=0.8, height=0.7)

        # labels: category name on left inside bar, count on right
        for i, (bar, cat, code, count) in enumerate(zip(bars, cats, codes, counts)):
            # count label outside bar
            ax.text(bar.get_width() + max(counts) * 0.008,
                    bar.get_y() + bar.get_height() / 2,
                    f'{count:,}',
                    va='center', ha='left', fontsize=9, fontweight='bold',
                    color='#333333')

            # arxiv code label inside bar (if bar wide enough)
            if bar.get_width() > max(counts) * 0.08:
                ax.text(bar.get_width() - max(counts) * 0.008,
                        bar.get_y() + bar.get_height() / 2,
                        code,
                        va='center', ha='right', fontsize=7,
                        color='white', fontstyle='italic', alpha=0.8)

        ax.set_yticks(range(len(cats)))
        ax.set_yticklabels(cats, fontsize=10)
        ax.set_xlabel('Number of Papers', fontsize=11, labelpad=10)
        ax.set_title('Pure Math Papers by arXiv Category',
                    fontsize=14, fontweight='bold', pad=20, loc='left')

        # subtitle
        total = sum(counts)
        ax.text(0, 1.02, f'{total:,} papers across {len(cats)} categories',
                transform=ax.transAxes, fontsize=10, color='#666666')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='y', length=0)
        ax.set_xlim(0, max(counts) * 1.12)

        # light gridlines
        ax.xaxis.grid(True, alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)

        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())

        print(f'Saved histogram at {save_path}')
        return save_path

    def parse_arxiv_manifest(self, xml_path=None, save_path=None):
        if xml_path is None:
            xml_path = './arXiv_src_manifest.xml'
        
        if save_path is None:
            save_path = 'src_manifest.json'

        tree = ET.parse(xml_path)
        root = tree.getroot()

        records_20yy = defaultdict(list)
        records_19yy = defaultdict(list)
        for element in root.findall("file"):
            first_item = element.findtext('first_item')
            yymm = element.findtext('yymm')

            if yymm.startswith('9'):
                records_19yy[yymm].append(first_item)
            else:
                records_20yy[yymm].append(first_item)

        with open(save_path, 'w') as f:
            records = {
                **records_19yy,
                **records_20yy
            }
            f.write(json.dumps(records, indent=2))

        return str(save_path)

    def find_src_batch(self, save_path=None, hash_table_path=None, metadata_dir=None):
        """
        Example arXiv ID '1011.1500'
        Hash table is produced by the parse_arxiv_manifest method
        """
        if hash_table_path is None:
            hash_table_path = './src_manifest.json'

        if save_path is None:
            save_path = self.save_dir / 'inventory.json'

        with open(hash_table_path, 'r') as f:
            hash_table = json.loads(f.read())

        histogram = defaultdict(int)
        i = 0
        for arxiv_id in self.arxiv_ids_iterator(dir_path=metadata_dir):
            i += 1
            file = self.find_src(arxiv_id, hash_table)
            if file is None:
                continue
            histogram[file] += 1
            if i % 100_000 == 0:
                print('Processed {i} arXiv IDs')

        with open(save_path, 'w') as f:
            f.write(json.dumps(histogram, indent=2))

        save_path = str(save_path)
        print(f'Saved to {save_path}')
        return save_path

    def find_src(self, arxiv_id, hash_table, return_compact=True):
        """
        Refer to find_src_batch method for hash table.
        """
        yymm = arxiv_id[:4]
        try:
            shards = hash_table[yymm]
        except KeyError:
            print(f'Key {yymm} not found.')
            return None
        array_id = bisect.bisect_right(shards, arxiv_id) - 1
        shard_id = str(array_id+1).zfill(3)
        if return_compact:
            remote_path = f'{yymm}_{shard_id}'
        else:
            remote_path = f'src/arXiv_src_{yymm}_{shard_id}.tar'
        return remote_path
    
    def arxiv_ids_iterator(self, dir_path=None)-> Iterator[str]:
        """
        Returns an iterator of arxiv_id
        """
        if dir_path is None:
            dir_path = self.save_dir / 'pure_math'
        jsonl_files = dir_path.glob('*.jsonl')
        file_id = 0
        for jsonl_path in jsonl_files:
            file_id += 1
            with open(jsonl_path, 'r') as f:
                for line in f:
                    record = json.loads(line.strip())
                    arxiv_id = record['id']
                    yield arxiv_id
    
    def pure_math_categories(self):
        return {
            **self.algebra_filters(),
            **self.geometry_filters(),
            **self.topology_filters(),
            **self.analysis_filters(),
            **self.discrete_math_filters()
        }

                
    def algebra_filters(self):
        return {
            'math.AC': 'Commutative Algebra',
            'math.CT': 'Category Theory',
            'math.KT': 'K Theory Homology',
            'math.OA': 'Operator Algebras',
            'math.QA': 'Quantum Algebra',
            'math.GR': 'Group Theory',
            'math.RA': 'Rings Algebras',
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
            'math.CA': 'Classical Analysis of ODEs',
            'math.CV': 'Complex Variables',
            'math.DS': 'Dynamical Systems',
            'math.FA': 'Functional Analysis',        
            'math.SP': 'Spectral Theory',
            'math.PR': 'Probability',
        }
    
    def discrete_math_filters(self):
        return {
            'math.CO': 'Combinatorics',
            'math.LO': 'Logic',
        }
    
    def excluded_math_filters(self):
        return {
            'math.GM': 'General Mathematics',
            'math.HO': 'History & Overview',
            'math.IT': 'Information Theory',
            'math.MP': 'Mathematical Physics',
            'math.NA': 'Numerical Analysis',
            'math.OC': 'Optimization & Control',
            'math.ST': 'Statistics Theory',
        }
    
if __name__ == '__main__':
    arxiv = ArXivMathDataPipeline()
    # arxiv.filter_math_papers()
    # arxiv.filter_math_by_category()
    # arxiv.plot_pure_math_histogram()
    # arxiv.parse_arxiv_manifest()
    # attn_all_u_need = '1706.03762' # src/arXiv_src_1706_010.tar
    # arxiv.find_src_batch(attn_all_u_need)
    arxiv.find_src_batch()
