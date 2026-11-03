"""Mutation tests for actual regressions found during the course audit."""
import sys,json,copy
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.visual_course_content import validate_contracts,authored_answers
ROOT=Path(__file__).resolve().parents[2]
@pytest.fixture
def contracts():return json.loads((ROOT/'visual-lab/course/contracts.json').read_text())
def test_authored_answers_cover_every_original_exercise():
 answers=authored_answers();assert len(answers)==80
 assert all(set(x)==set(range(1,21))for x in answers.values())
 assert len({a for ch in answers.values()for a in ch.values()})==1600
@pytest.mark.parametrize('defect',['missing-answer','wrong-topic','missing-section','wrong-evidence','generic-rubric'])
def test_builder_rejects_a_broken_teaching_contract(contracts,defect):
 c=contracts['14'];e=c['exercises'][3]
 if defect=='missing-answer':e['answer']=''
 if defect=='wrong-topic':e['answerTopicIds']=['knn-voting']
 if defect=='missing-section':e['sections']=['not-a-section']
 if defect=='wrong-evidence':c['evidence'][0]['objectiveId']='ch08-objective-01'
 if defect=='generic-rubric':e['answer']='A complete answer includes input checks, quality monitoring and reproducibility metadata.'
 with pytest.raises(ValueError):validate_contracts(contracts)
def test_all_contracts_are_valid(contracts):assert validate_contracts(contracts)
def test_hints_reveal_progressively_without_repeating_or_dumping_the_full_solution(contracts):
 for ch in contracts.values():
  for exercise in ch['exercises']:
   if not exercise['number']:continue
   assert len(set(exercise['hints']))==3
   assert exercise['answer']not in exercise['hints']
 assert all(e['hintSource']=='authored'for e in contracts['6']['exercises']if e['number'])
def test_every_objective_has_a_versioned_evidence_notebook():
 import re
 contracts=json.loads((ROOT/'visual-lab/course/contracts.json').read_text())
 for n,ch in contracts.items():
  page=next(ROOT.glob(f'{int(n):02d}-*/animation.html')).read_text()
  assert len(re.findall(r'data-evidence="',page))==len(ch['evidence'])
  assert page.count('data-evidence-text=')==4*len(ch['evidence'])
  for evidence in ch['evidence']:assert f'data-evidence="{evidence["id"]}"'in page
def test_dbscan_answer_is_density_clustering_not_class_voting(contracts):
 e=contracts['14']['exercises'][3]
 assert e['sections']==['topic-11','topic-12']
 assert 'border' in e['answer']and'noise'in e['answer']and'core'in e['answer']
 assert 'KNN'not in e['answer'].split('ไม่มีการโหวต')[0]
def test_bootstrap_solution_is_resampling_not_wilson(contracts):
 answer=contracts['60']['exercises'][5]['answer']
 assert 'replacement'in answer and 'percentile'in answer
 assert 'ไม่ใช้Wilson'in answer
def test_regression_sections_have_specific_explanations_and_verified_numbers():
 teaching=json.loads((ROOT/'visual-lab/course/authored/regression-teaching.json').read_text())
 assert 'Δŷ=x·Δw'in teaching['slope']
 assert 'ทุกจุดขึ้น .2 เท่ากัน'in teaching['bias']
 assert 'รวม15และหาร5ได้MSE3'in teaching['residual-mse']
 assert 'mean(2ex)'in teaching['derivation']
 assert '1.36'in teaching['gradient-descent']
 assert '0<η<.5'in teaching['loss-surface']
 import numpy as np
 x=np.arange(-2,3);y=2*x+1;w,b=1.,0.;lam=.2
 g=2*np.mean((w*x+b-y)*x)+2*lam*w;gb=2*np.mean(w*x+b-y)
 w-=.1*g;b-=.1*gb
 assert np.mean((w*x+b-y)**2)+lam*w*w==pytest.approx(1.82912)
def test_required_experiments_cannot_be_misrouted():
 from tools.build_visual_contracts import compile_contracts
 chapters=json.loads((ROOT/'visual-lab/course/chapters.json').read_text())
 for lab in chapters['14']['labs']:
  if lab['model']=='dbscan':lab['model']='neighbors'
 with pytest.raises(ValueError,match='dbscan'):compile_contracts(chapters)
def test_real_framework_and_pca_shapes_agree_with_assessments():
 import numpy as np
 import torch
 from sklearn.decomposition import PCA
 from tools.visual_course_shapes import case
 for v in range(3):
  n,d,out=2+v,4+v,5+v
  layer=torch.nn.Linear(d,out);actual=tuple(layer(torch.zeros(n,d)).shape)
  assert case(22,v)[2]==str(actual).replace(' ','')
  actual=PCA(n_components=1+v).fit_transform(np.random.default_rng(7).normal(size=(n,d))).shape
  assert case(15,v)[2]==str(actual).replace(' ','')
def test_gpu_grid_questions_cover_exact_and_partial_blocks():
 from tools.visual_course_shapes import case
 for v,n in enumerate([1023,1024,1025]):
  grid=int(case(53,v)[2]);assert grid*256>=n and (grid-1)*256<n
