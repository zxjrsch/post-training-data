import bisect
import gzip
import io
import json
import re
import shutil
import tarfile
import traceback
import uuid
import xml.etree.ElementTree as ET
from collections import defaultdict
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Iterator, List, Optional

import boto3
import matplotlib.pyplot as plt
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi
from loguru import logger
from omegaconf import OmegaConf
from tqdm import tqdm


@dataclass
class ExtractionResult:
    """
    Result of extracting LaTeX documents from an arXiv bundle like s3://arxiv/pdf/arXiv_pdf_1001_003.tar
    """
    serialized_documents: str
    missing_arxiv_ids: List[str]
    pdf_arxiv_ids: List[str]    # PDF found, but not LaTeX source available
    

class ArXivMathDataPipeline:
    def __init__(self, config_yaml_path: str|Path):

        cfg = OmegaConf.load(config_yaml_path)

        self.metadata_path = Path(cfg.metadata_path)
        self.save_dir = Path(cfg.save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        self.inventory_json = Path(cfg.inventory_json)
        self.remove_zip = cfg.remove_zip
        self.hf_auto_upload = cfg.hf_auto_upload
        self.parquet_output_dir = Path(cfg.parquet_output_dir)
        self.parquet_output_dir.mkdir(parents=True, exist_ok=True)
        self.approx_papers_per_parquet = cfg.approx_papers_per_parquet
        self.progress_stats_dir = Path(cfg.progress_stats_dir)
        self.progress_stats_dir.mkdir(exist_ok=True, parents=True)

        self.run_id = uuid.uuid4()
        logger.add(self.progress_stats_dir / f'run_{self.run_id}.log')
        logger.info(f'Run {self.run_id}')

        if cfg.authenticate:
            self.s3 = boto3.client("s3")
            if 'hf_token' in cfg and 'hf_user' in cfg and 'repo_name' in cfg:
                logger.info(f"Creating HF data repo.")
                self.hf = HfApi(token=cfg.hf_token)
                self.repo_id = f"{cfg.hf_user}/{cfg.repo_name}"
                self.hf.create_repo(self.repo_id, repo_type="dataset", private=True, exist_ok=True)
            else:
                logger.warning(f"hf_token or hf_user or repo_name is not set in {config_yaml_path}")

            
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
                if math_papers % log_interval == 0:

                    # stats
                    percentage = round(100 * math_papers/total_papers, 2)
                    dt = perf_counter() - t0

                    logger.info(f'Found {math_papers} math papers | {percentage}% total papers | this shard took {dt} seconds')
                    t0 = perf_counter()
            out_f.write(''.join(papers))
            dt_min = round((perf_counter()-start)/60, 1)
            logger.info(f'End. Found {math_papers} math papers. Took {dt_min} min')

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
                    logger.info(f'Processed {i} math records | latest 100_000 records took {dt} seconds.')
                    t0 = perf_counter()

            dt = round(perf_counter()-start_t)

            # stats 
            percent_excluded = round(100*num_excluded_records/i, 1)
            logger.info(f'Complete in {dt} seconds. Saved {i-num_excluded_records} records. Excluded {num_excluded_records} ({percent_excluded}%) records.')
        
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

    #     logger.info(f'Saved histogram at {save_path}')
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

        logger.info(f'Saved histogram at {save_path}')
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
    
    def sort_inventory(self, from_most_recent=True):
        """"Sort yymm"""
        with open(self.inventory_json, 'r') as f:
            inventory = json.loads(f.read())
            inventory = dict(sorted(inventory.items(), key=lambda item: item[0], reverse=True))
        with open(self.inventory_json, 'w') as f:
            f.write(json.dumps(inventory, indent=2))
        logger.info(f'Sorting complete | from_most_recent {from_most_recent} | path: {str(self.inventory_json)}')
            
    def verify_filename_validity(self, xml_path=None, inventory_path=None):
        if xml_path is None:
            xml_path = './arXiv_src_manifest.xml'
        if inventory_path is None:
            inventory_path = self.inventory_json

        with open(inventory_path, 'r') as f:
            inventory = json.loads(f.read())

        tree = ET.parse(xml_path)
        root = tree.getroot()

        all_files = defaultdict(str)
        for element in root.findall("file"):
            filename = element.findtext('filename')
            yymm = element.findtext('yymm')
            all_files[filename] = yymm

        for shorthand in tqdm(inventory.keys(), desc=f'Asserting filename validity'):
            assert f'src/arXiv_src_{shorthand}.tar' in all_files

    def find_src_batch(self, save_dir=None, hash_table_path=None, metadata_dir=None):
        """
        Example arXiv ID '1011.1500'
        Hash table is produced by the parse_arxiv_manifest method
        """
        if hash_table_path is None:
            hash_table_path = './src_manifest.json'

        if save_dir is None:
            inventory_count_path = self.save_dir / 'inventory_count.json'
            inventory_path = self.inventory_json

        with open(hash_table_path, 'r') as f:
            hash_table = json.loads(f.read())

        histogram = defaultdict(int)
        inventory = defaultdict(list)
        i = 0
        for arxiv_id in self.arxiv_ids_iterator(dir_path=metadata_dir):
            i += 1
            file = self.find_src(arxiv_id, hash_table)
            if file is None:
                continue
            histogram[file] += 1
            inventory[file].append(arxiv_id)
            if i % 100_000 == 0:
                logger.info(f'Processed {i} arXiv IDs')

        with open(inventory_count_path, 'w') as f:
            f.write(json.dumps(histogram, indent=2))

        with open(inventory_path, 'w') as f:
            f.write(json.dumps(inventory, indent=2))

        logger.info(f'Done, processed {i} arXiv IDs')
        inventory_path = str(inventory_path)
        logger.info(f'Saved to {inventory_path}')
        return inventory_path

    def find_src(self, arxiv_id, hash_table, return_compact=True):
        """
        Refer to find_src_batch method for hash table.
        """
        yymm = arxiv_id[:4]
        try:
            shards = hash_table[yymm]
        except KeyError:
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
    
    def ls(self, prefix='src/', max_entries=10):
        """
        List files from arXiv S3 bucket
        """
        try:
            response = self.s3.list_objects_v2(
                Bucket='arxiv',
                Prefix=prefix,
                Delimiter="/",
                MaxKeys=max_entries,
                RequestPayer='requester'
            )
            for p in response.get("CommonPrefixes", []):
                logger.info(f"  [DIR]  {p['Prefix']}")
            for obj in response.get("Contents", []):
                logger.info(f"  [FILE] {obj['Key']}  ({obj['Size']} bytes)")

        except Exception as e:
            logger.info(f"Error: {e}")
    
    def process_shard(self, skip: int, num_arxiv_zip_files: int, inventory_path: Optional[str|Path]=None):
        """
        Downloads shard_len source zipped files from arXiv s3 and export as parquet, supports huggingface upload
        Ensure approximately equal number of papers by assigning appropriate shard_len based on self.inventory_json
        """
        assert skip >=0 and num_arxiv_zip_files > 0
        
        if inventory_path is None:
            inventory_path = self.inventory_json
        with open(inventory_path, 'r') as f:
            inventory = json.loads(f.read())
            inventory = iter(inventory.items())

        start_zip = skip+1
        end_zip = skip + num_arxiv_zip_files
        file_name = f'arxiv-math-{start_zip:04d}-{end_zip:04d}-shard-1'
        local_parquet_path = self.parquet_output_dir / f'{file_name}.parquet'
        remote_parquet_path = f'data/{file_name}.parquet'
        stats_path = self.progress_stats_dir / f'{file_name}.json'

        for _ in range(skip):
            next(inventory)

        stats = {'run_id': self.run_id, 'start_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        shard_start_time = perf_counter()
        try:
            writer = None
            num_papers_current_shard = 0
            shard_id = 1
            for _ in range(num_arxiv_zip_files):
                if num_papers_current_shard > self.approx_papers_per_shard:
                    if writer:
                        writer.close()
                    if self.hf_auto_upload:
                        self.upload_parquet(local_parquet_path=local_parquet_path, remote_parquet_path=remote_parquet_path)
                        stats[f'shard_{shard_id}'] = {
                            'upload_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'num_papers': num_papers_current_shard,
                            'remote_file_path': remote_parquet_path
                        }
                    # refresh
                    writer = None
                    num_papers_current_shard = 0
                    shard_id += 1
                    file_name = f'arxiv-math-{start_zip:04d}-{end_zip:04d}-shard-{shard_id}'
                    local_parquet_path = self.parquet_output_dir / f'{file_name}.parquet'
                    remote_parquet_path = f'data/{file_name}.parquet'

                k, v = next(inventory)
                num_papers_current_shard += len(v)
                t0 = perf_counter()
                res = self.download_and_uzip(filename_short=k, unzip_arxiv_ids=v, shard=shard_id, delete_zip=self.remove_zip, delete_unzipped=True)
                dt = perf_counter() - t0
                if res is not None:
                    logger.info(k)
                    table = pa.table({'arXiv_src_id': [k], 'serialized_document_string': [res.serialized_documents]})
                    if writer is None:
                        writer = pq.ParquetWriter(local_parquet_path, table.schema)
                    writer.write_table(table)

                    # record stats
                    stats[k] = {
                        'status': 'ok', 
                        'download_and_uzip took time (seconds)': str(dt),
                        'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
                        'missing_arxiv_ids': ','.join(res.missing_arxiv_ids), 
                        'pdf_arxiv_ids': ','.join(res.pdf_arxiv_ids),
                        'requested_arxiv_ids': ','.join(v),
                    }
                else:
                    stats[k] = {'status': 'error', 'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'requested_arxiv_ids': ','.join(v)}
        finally:
            if writer:
                writer.close()

            with open(stats_path, 'w') as f_stats:
                stats['end_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                stats['shard_processing_time'] = perf_counter() - shard_start_time
                f_stats.write(json.dumps(stats, indent=2))

        if self.hf_auto_upload:
            self.upload_parquet(local_parquet_path=local_parquet_path, remote_parquet_path=remote_parquet_path)
     
        return str(local_parquet_path)
    
    def inspect_parquet(self, path=None):
        if path is None:
            path = next(self.parquet_output_dir.rglob('*.parquet'))
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=1):
            row = batch.to_pydict()
            logger.info(row['arXiv_src_id'][0])
            for record in row['serialized_document_string'][0].strip().split('\n'):
                record = json.loads(record)
                logger.info(record['arxiv_id'])
                # for latex_file in [k for k in record.keys() if k != 'arxiv_id']:
                #     logger.info(record[latex_file])
                #     break
                

    def download_and_uzip(self, filename_short: str, unzip_arxiv_ids: list[str], shard: Optional[int]=None, delete_zip=False, delete_unzipped=True):
        """
        Short filename example 1706_010 yymm_zzz where zzz is the zip file number.
        unzip_arxiv_ids are arXiv IDs to be extracted from this zip file 
        """
        local_base = self.save_dir / 'download'
        if shard is not None:
            local_base = local_base / f'shard_{shard}'
        
        local_base.mkdir(parents=True, exist_ok=True)
        
        remote_filename = f'src/arXiv_src_{filename_short}.tar'
        local_zip_path = local_base / f'{filename_short}.tar'
        extracted_path = local_base / f'{filename_short}'

        try:
            t0 = perf_counter()
            if not local_zip_path.exists():
                # logger.info(f'Downloading {remote_filename}')
                self.s3.download_file(
                    Bucket="arxiv",
                    Key=remote_filename,
                    Filename=str(local_zip_path),
                    ExtraArgs={"RequestPayer": "requester"},
                )
                t1 = perf_counter()
                dt = round(t1-t0)
                logger.info(f'Downloading {remote_filename} took {dt} seconds.')
                t0 = t1

            result = self.extract_latex_from_zip(
                local_zip_path=local_zip_path, 
                extracted_path=extracted_path, 
                unzip_arxiv_ids=unzip_arxiv_ids
            )
            t1 = perf_counter()
            dt = round(t1-t0)
            logger.info(f'Decompressing & extracting {remote_filename} took {dt} seconds.')

            if delete_unzipped:
                shutil.rmtree(extracted_path, ignore_errors=True)
            if delete_zip:
                local_zip_path.unlink(missing_ok=True)
            return result

        except Exception as e:
            traceback.logger.info_exc()
            return None
        
    def extract_latex_from_zip(self, local_zip_path: str|Path, extracted_path: str|Path, unzip_arxiv_ids: List[str]):
        """
        Returns ExtractionResult 
        """
        local_zip_path = Path(local_zip_path)
        extracted_path = Path(extracted_path)
        extracted_arxiv_ids = set()
        unzip_arxiv_ids = set(unzip_arxiv_ids)
        pdf_arxiv_ids = []   # arXiv id found in zip but latex source not found
        with tarfile.open(local_zip_path) as tar:
            paper_json_lines = []
            # logger.info(tar.getnames())
            for gz_path in tar.getnames():
                arxiv_id = re.sub(r'v\d+$', '', Path(gz_path).stem)
                arxiv_id = str(arxiv_id)
                if arxiv_id in unzip_arxiv_ids:
                    try:
                        f = tar.extractfile(gz_path)
                        compressed_content = f.read()
                        if compressed_content[:5] == b'%PDF-':
                            extracted_arxiv_ids.add(arxiv_id)
                            pdf_arxiv_ids.append(arxiv_id)
                            continue
                        # at this point it should be a valid zip or single LaTeX file
                        try:
                            f_content = gzip.decompress(compressed_content)
                        except gzip.BadGzipFile:
                            # single file submission, not gzipped
                            f_content = compressed_content
                        # extract paper
                        extracted_paper_path = extracted_path / arxiv_id
                        try:
                            with tarfile.open(fileobj=io.BytesIO(f_content)) as inner_tar:
                                inner_tar.extractall(path=extracted_paper_path, filter='data')
                        except tarfile.ReadError:
                            # gzipped single .tex file, not a tarball
                            files = {
                                'arxiv_id': arxiv_id,
                                'main.tex': f_content.decode('utf-8', errors='replace'),
                            }
                            line = json.dumps(files) + '\n'
                            paper_json_lines.append(line)
                            extracted_arxiv_ids.add(arxiv_id)
                            continue

                        line = self.get_paper_latex(dir_path=extracted_paper_path, arxiv_id=arxiv_id)  # returns serialized string
                        if line is None:
                            continue
                        paper_json_lines.append(line)
                        extracted_arxiv_ids.add(arxiv_id)
                    except Exception as e:
                        logger.info(f'Error processing arXiv:{arxiv_id} | Local zip path {str(local_zip_path)} | Extracted path {str(extracted_path)}')
                        traceback.logger.info_exc()

        missing_arxiv_ids = list(unzip_arxiv_ids - extracted_arxiv_ids)
        # logger.info('arXiv IDs not found:', missing_arxiv_ids, 'PDF exists but no LaTeX:', pdf_arxiv_ids)
        serialized_documents = ''.join(paper_json_lines)
        result = ExtractionResult(serialized_documents=serialized_documents, missing_arxiv_ids=missing_arxiv_ids, pdf_arxiv_ids=pdf_arxiv_ids)
        return result
        
    def get_paper_latex(self, dir_path, arxiv_id):
        """
        Iterates through a source directory and extracts LaTeX files. Outputs a serialized dict
        with a key named "arxiv_id" and other keys being the relative file paths and values being file content.
        """
        latex_paths = list(Path(dir_path).rglob('*.tex'))
        if not len(latex_paths):
            return None
        files = {'arxiv_id': arxiv_id}
        for path in latex_paths:
            k = str(path.relative_to(dir_path))
            with open(path, 'r', errors='replace') as f:
                files[k] = f.read().strip()
        return json.dumps(files) + '\n' 

    
    def upload_folder(self, folder_path=None):
        if folder_path is None:
            folder_path = self.parquet_output_dir
        assert self.repo_id
        self.hf.upload_folder(
            folder_path=folder_path,
            repo_id=self.repo_id,
            repo_type="dataset",
            path_in_repo="data",
        )

    def upload_parquet(self, local_parquet_path, remote_parquet_path):
        assert self.repo_id
        self.hf.upload_file(
            path_or_fileobj=local_parquet_path,
            path_in_repo=remote_parquet_path,
            repo_id=self.repo_id,
            repo_type="dataset",
        )

    def divide_workload(self, num_cpus: int=64, workload_path = 'workload.jsonl'):
        """
        The current filter has 478567 papers. Suppose there are 8 CPUs (8 workers).
        Then each processes 60k papers. If each shard is 10k (~500 MB), each worker creates 6 shards (3GB).
        """
        PROCESSING_TIME_PER_ZIP = 3 # seconds

        assert num_cpus > 0
        
        logger.info(f'Calculating workload share...')
        with open(self.inventory_json, 'r') as f:
            inventory = json.loads(f.read())
        total_papers = 0
        for k, v in inventory.items():
            total_papers += len(v)
        papers_per_cpu = total_papers // num_cpus

        min_zips_per_cpu = float('inf')
        max_zips_per_cpu = -float('inf')

        workload = []
        skip = num_arxiv_zip_files = 0
        cpu = current_cpu_papers = 0
        current_cpu_zips = []
        for k, v in inventory.items():

            num_arxiv_zip_files += 1
            current_cpu_zips.append(k)
            current_cpu_papers += len(v)

            if current_cpu_papers > papers_per_cpu:
                max_zips_per_cpu = max(max_zips_per_cpu, num_arxiv_zip_files)
                min_zips_per_cpu = min(min_zips_per_cpu, num_arxiv_zip_files)
                job = {
                    'cpu': cpu,
                    'skip': skip,
                    'num_arxiv_zip_files': num_arxiv_zip_files,
                    'current_cpu_zips': ','.join(current_cpu_zips)
                }
                workload.append(json.dumps(job)+'\n')
                cpu += 1
                skip += num_arxiv_zip_files
                num_arxiv_zip_files = 0
                current_cpu_papers = 0
                current_cpu_zips = []

        if workload:
            job = {
                'cpu': cpu,
                'skip': skip,
                'num_arxiv_zip_files': num_arxiv_zip_files,
                'current_cpu_zips': ','.join(current_cpu_zips)
            }
            workload.append(json.dumps(job))

        with open(workload_path, 'w') as workload_f:
            workload_f.write(''.join(workload))

        logger.info(f'Divided work into {len(workload)} parts, saved to {str(workload_path)}.')
        num_parquets_per_cpu = (papers_per_cpu + self.approx_papers_per_parquet-1) // self.approx_papers_per_parquet
        last_shard = papers_per_cpu % self.approx_papers_per_parquet
        logger.info(f'Total papers {total_papers} | CPUs {num_cpus} | {papers_per_cpu} papers/cpu | {num_parquets_per_cpu} parquet/cpu | {self.approx_papers_per_parquet} papers/shard | Last shard has {last_shard} papers')   
        
        min_t = round(PROCESSING_TIME_PER_ZIP * min_zips_per_cpu / 60, 1)
        max_t = round(PROCESSING_TIME_PER_ZIP * max_zips_per_cpu / 60, 1)

        logger.info(f'Estimated Finish Time: {min_t} - {max_t} min.')
    
if __name__ == '__main__':
    arxiv = ArXivMathDataPipeline(config_yaml_path='config.yaml')
    # arxiv.filter_math_papers()
    # arxiv.filter_math_by_category()
    # arxiv.plot_pure_math_histogram()
    # arxiv.parse_arxiv_manifest()
    # attn_all_u_need = '1706.03762' # src/arXiv_src_1706_010.tar
    # arxiv.find_src_batch(attn_all_u_need)
    # arxiv.find_src_batch()
    # arxiv.verify_filename_validity()
    # arxiv.extract_latex_from_zip(
    #     local_zip_path='./tar/arXiv_src_1706_010.tar',
    #     extracted_path='./tar/arXiv_src_1706_010',
    #     unzip_arxiv_ids=['1706.03762', '1706.03627', '1234.5678']
    # )
    # arxiv.download_and_uzip()

    # arxiv.process_shard()
    # arxiv.inspect_parquet()
    # arxiv.sort_inventory()
    arxiv.divide_workload()
