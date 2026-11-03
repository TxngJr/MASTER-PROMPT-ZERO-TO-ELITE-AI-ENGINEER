# เฉลยเฉพาะข้อ: model families
## 34
1. sentencepair “I code”/“AI works”จัดเป็น[CLS],I,code,[SEP],AI,works,[SEP]; specialIDsต้องมาจากtokenizerเดียวกับencoder และattentionlength7
2. token-typeIDsของข้อ1คือ[0,0,0,0,1,1,1]; firstSEPเป็นsegment0,secondSEPsegment1; บางencoderไม่ใช้token-typeจึงต้องตรวจarchitecture
3. `eligible=~np.isin(ids,special_ids); selected=(rng.random(len(ids))<.15)&eligible`; labelsของunselectedเป็นignore_index−100; assertspecialไม่ถูกเลือก
4. สำหรับselectedpositionsใช้u~Uniform: u<.8แทนMASK,.8≤u<.9แทนrandomtoken,ที่เหลือคงเดิม; 10000drawsควรใกล้80/10/10แต่ไม่บังคับตรงจำนวนทุกseed
5. ids=[CLS,a,SEP,PAD,PAD]ให้validmask[1,1,1,0,0]; keypaddingmaskต้องปิดPADkeyทุกquery และlossmaskแยกต่างหาก
6. logits(B,T,V)เฉพาะselectedpositions; ถ้าselectedtrueที่1,3ให้CE=(CE1+CE3)/2; selected0ตำแหน่งต้องskipbatchหรือerrorไม่meanemptyเป็นNaN
7. CLSpoolingใช้hidden[:,0,:]; maskedmean=sum(hidden*valid[...,None])/sum(valid)และกำหนดรวมspecialหรือไม่; paddingมากขึ้นไม่ควรเปลี่ยนmeanของcontentเดิม
8. `logits=hidden@W+b` โดยhidden(B,T,d),W(d,C)→(B,T,C); CEเฉพาะvalidtokenและlabelสำหรับsubwordต้องกำหนดfirst-subword/all-subwordpolicy
9. seed7syntheticlabelจากkeyword,splitdocuments,trainencoderheadแล้วreportvalaccuracyกับmajoritybaseline; ผลผ่านต้องtestlossfinite/headgradnonzeroและไม่มีexampleleak
10. MLMเห็นซ้ายและขวาระหว่างทำนายmaskedtoken จึงไม่ใช่P(next|prefix)ตามcausalfactorization; จะgenerateต้องเปลี่ยนmask/objectiveหรือใช้อีกsamplingprocedureที่ระบุ
11. exampleMLMprobtrue.25และ.5สองselectedtokensให้loss(−log.25−log.5)/2=1.039721; unselectedlogitsเปลี่ยนแล้วlossต้องคงเดิม
12. testselectedmaskไม่โดนCLS/SEP/PADและunselectedlabels−100; appendPADแล้วvalidloss/pooledcontentต้องallcloseในevalmode
13. เปรียบCLSpoolกับmaskedmeanบนsplit/weights/seedเดียวและreportaccuracy/F1; trainheadsงบเท่ากันและนิยามspecialtokenmaskชัดเจน
14. recordids(B,T),mask(B,T),hidden(B,T,d),logits(B,C),rangesfiniteและmacroF1; ใช้Tต่างกันในbatchเพื่อจับpaddingbug
15. ใช้sentencepairduplicateข้ามtrain/testหรือselectcheckpointด้วยtestทำคะแนนสูงปลอม; splitโดยdocument/sourceก่อนtokenizationและkeepheldoutuntouched
16. encoderattentionO(BT²d),FFNO(BTddff),scoresO(BhT²); MLMheadfullvocabO(BTVd)แม้selectedน้อยหากคำนวณทุกtoken
17. เทียบmanualselectedCEกับtorchCrossEntropyLoss(ignore_index=−100,reduction='mean')บนlogitsเดียวกัน; evaldropout0และtokenizer/specialIDsตรง
18. เปลี่ยนmaskrate.15→.30คงcorpus/seed/budget; reportMLMlossบนfixedvalidationmaskและclassificationF1หลายrunsไม่เทียบlossจากmaskคนละชุด
19. manifestต้องมีcorpushash,splitIDs,tokenizerversion,specialIDs,maskseed/rate/corruptionpolicy,encodercheckpoint,headconfig,commit,metricdefinition
20. สมมติฐาน“maskedmeanเพิ่มmacroF1≥.02บนlongsentencesโดยlatency≤1.1×CLS”; baselineCLS,เปลี่ยนpoolingอย่างเดียว,pairedholdoutพร้อมCI
## 35
1. tokens[2,5,7,9]ทำinput[2,5,7],target[5,7,9]; ไม่ใช้targetเท่ากับinputเพราะmodelเพียงcopycurrenttokenก็lossต่ำ
2. T6keepmaskแถว0..5คือ100000,110000,111000,111100,111110,111111; futurej>iเป็น−∞ก่อนsoftmax ไม่ใช่0logit
3. `next_id=np.argmax(logits[-1]); ids.append(int(next_id))` วนจนEOS/max_new_tokens; ไม่sampleและtieใช้indexแรกตามpolicy
4. logits[0,log3],T1prob[.25,.75],T.5prob[.1,.9],T2prob≈[.366025,.633975]; T>0และไม่แก้weights
5. topk2ของlogits[3,2,1]เก็บindices0,1และsetindex2−∞ก่อนsoftmax; k≤0หรือk>Vต้องreject/กำหนดนโยบายชัด
6. probs[.6,.3,.1],p.7เก็บสองtokenแรกเพราะcumulativeข้าม.7ที่ตัวสอง; ต้องเก็บtokenที่ทำให้ข้ามthresholdและอย่างน้อย1token
7. แบ่งlogitsด้วยT→topkfilter→toppfilter→renormalize→sampleด้วยseed; การเรียงfilterมีผลจึงต้องบันทึกpolicy
8. greedyrepeatableแต่ไม่สำรวจdistribution,stochasticsampleให้diversityและอาจผิดมากขึ้น; ประเมินquality/diversityหลายseedและbudgetเท่ากัน
9. KVแต่ละlayer shape(B,nkv,Tcache,dk); เพิ่มtokenหนึ่งทำTcache+1,K/Vเก่าคงเดิมและqueryใหม่length1; ไม่appendqueryเข้าcache
10. bytes=2·L·B·nkv·T·dk·bytes_per_element; L2,B1,nkv2,T16,dk4,fp32ได้2048bytesไม่รวมweights/allocator
11. correctnexttokenprob[.5,.25]สองตำแหน่งCE1.039721,PPLexp(CE)=2.828427; PADไม่อยู่denominatorและtokenizerต้องตรงตอนcompare
12. cached/uncachedlogitsallcloseทุกstepในevaldropout0; maskfuture0และcachelengthเพิ่ม1; testprefixlength2กับpositionoffset2จับRoPEoffbyone
13. คงprompt/model/seed เปรียบT.5/1ด้วยtopk/toppเดิม; recordentropy/repetitionและtaskaccuracy ห้ามอ้างprobเปลี่ยนคือmodelretrained
14. recordids(B,T),Q(B,h,Tq,dk),cacheK/V,logits(B,Tq,V),peakbytesและtokenNLL; prefillกับdecodeแยกstage
15. overlappingdocumentwindowsข้ามsplitทำPPLต่ำปลอม; reportdedup/tokenizer/lengthและไม่รวมprompt/paddingในcompletionmetricโดยไม่บอก
16. fullprefillO(T²d),singlecacheddecodeattentionO(Td)แต่KVmemoryO(LTdnkv/h); naiveทั้งprefixซ้ำแพงกว่าและweightsprojectioncostยังมี
17. copytinydecoderweightsไปtorchmanualcausalattention,dropout0; compareprefillและperstepcachedlogitsatol1e−6พร้อมsametokenIDs
18. cacheon/offคงgeneratedtokenpathด้วยgreedyเพื่อเทียบnumerical/latency; warmupและจับprefill/decodeแยกพร้อมcontextlengthหลายค่า
19. บันทึกcheckpoint/tokenizer/promptIDs/T/k/p/EOS/seed/cachelayout/dtype/positionoffset/commitและhardware; hashlogitsfixtureช่วยตรวจregression
20. ทดลอง“GQAลดKVbytes4×โดยqualityลด≤1จุด”; baselineMHA,เปลี่ยนheadsharingพร้อมtrainingbudgetเท่ากัน,วัดholdoutquality/TPOT/bytes
## 36
1. encoderX(B,S,d)→H(B,S,d); decoderY(B,T,d); crossQ(B,h,T,dk),K/V(B,h,S,dk),scores(B,h,T,S),output(B,T,d)
2. target[a,b,EOS]ให้decoderinput[BOS,a,b],labels[a,b,EOS]; PADlabelsใช้ignore_index ไม่ให้BOSกลายเป็นtargetแรกโดยผิดshift
3. source[a,b,PAD]valid[1,1,0],target[BOS,a,PAD]valid[1,1,0]; sourcepaddingmaskใช้กับencoderselfและcrosskey,targetmaskใช้กับdecoderself/loss
4. encoderselfS×Sbidirectionalเฉพาะvalidkeys; decoderselfT×Tcausal+targetpadding; crossT×Svalidsourcekeysไม่ใช้causalตามsourceindex
5. `A=softmax(Q@K.T/np.sqrt(dk)+source_mask,axis=-1); O=A@V`; T2,S3ให้A(2,3)แม้Q/Klengthไม่เท่ากัน
6. original[a,b,c,d,e]maskspan[b,c]และ[e]ให้source[a,X0,d,X1],target[X0,b,c,X1,e,X2]ตามsentinel-endpolicyที่ประกาศ
7. สุ่มstart/lengthบนeligibleindicesแล้วmarkoccupiedก่อนเลือกspanถัดไป; assertไม่มีoverlapและdecodeใส่spanคืนได้เดิม; seedตรึงผลได้
8. seed7tokensvocab8,traincopy/reverseโดยsplitsequences; exact-sequencematchเทียบcopybaselineและrecordteacherforcedlossแยกfreerunaccuracy
9. greedyเลือกlocalmax,beamเก็บหลายprefixตามcumulativelogprob; beamอาจชอบsequenceสั้นจึงต้องlengthpenaltyและไม่ได้รับประกันqualityดีกว่า
10. BERT:bidirectionalMLM→maskedtokens; GPT:causalnexttoken→continuation; T5:encoderinputcorrupted+causaldecodertargetspans→conditionalgeneration
11. crossattentionQ=[1,0],K=[[1,0],[0,1]],V=[[2,0],[0,4]],dk2ให้O≈[1.339523,1.320954]จากweights.669762/.330238
12. เปลี่ยนpaddedsourcetokenembeddingแล้วvaliddecoderoutputsต้องคงเดิม; เปลี่ยนfuturetargetห้ามกระทบearlierdecoderoutputในevalmode
13. คงsyntheticreverse/split/model เปรียบgreedyกับbeam2โดยEOS/maxlengthเท่ากัน; reportexactmatchและdecodecomputecostจริง
14. recordencoder(B,S,d),crossscores(B,h,T,S),decoderlogits(B,T,V),maskmass0และsequenceexactmatch; S≠Tช่วยจับwrongmask
15. teacherforcedtokenaccuracyสูงอาจซ่อนfreerunningerrorสะสม; ต้องgenerateโดยไม่ป้อนgroundtruthและไม่ปล่อยtargetเข้าencoder
16. attentionO(S²d+T²d+STd)ต่อpairedblocks,cacheencoderK/Vช่วยไม่projectsourceซ้ำทุกdecode; beamเพิ่มbatchและcachememory
17. เทียบNumPycrossattentionกับtorchfunctionalSDPAด้วยsourcepaddingmask/dropout0,source/targetlengthต่างกันและweightsidentical
18. เปลี่ยนspanlengthmean2→3คงmaskfraction/corpus/updates; fixedvalidationcorruptionsและreportreconstructionexactmatch/NLLพร้อมหลายseed
19. manifestเก็บsource/targettokenizer,sentinelIDs,shift/BOS/EOS/maskrules,splitsequencehash,beamconfig,seed,checkpoint,metric
20. ทดลอง“crossattentionจำเป็นต่อreverseเมื่อS>trainlength”; baselinefullseq2seq,ablateencoderconnection,วัดexactmatchแยกlengthsและlatency
## 37
1. edges(0,1),(1,2)undirectedทำA=[[0,1,0],[1,0,1],[0,1,0]]; directedต้องไม่ใส่reverseedgeเองและduplicateedgesต้องกำหนดsum/binary
2. addselfloopให้degrees[2,3,2]; Ahat=D^−½(A+I)D^−½ มีdiag[.5,1/3,.5],neighborweights1/√6≈.408248 ไม่ใช่rowmean
3. neighborsvectors[1,3]sum4,mean2,max3; emptynodeกำหนดzeroหรือselfloop; meanหารdegreeจริงไม่หารจำนวนnodeทั้งหมด
4. permutationneighbororderไม่เปลี่ยนsum/mean/maxเพราะcommutative; testnp.array([1,3,2])ทุกpermutationได้sum6,mean2,max3
5. `H1=np.maximum(Ahat@X@W1,0); H2=Ahat@H1@W2`; X(3,d),W1(d,h),W2(h,c)ให้H2(3,c)และใช้Ahatเดิมทั้งสองlayer
6. GraphSAGEmean hnew=φ(W concat(hself,mean(hneighbors))); selfกับneighborsแยกก่อนconcat ไม่ใช้GCNnormalizationแทนโดยไม่บอก
7. GAT eij=LeakyReLU(aᵀ[Whi||Whj]),normalizeเฉพาะneighbors+jselfตามpolicy; nonedgeattention0 และแต่ละrowvalidsum1
8. identityweights/repeatedAhatมักทำnodefeaturesคล้ายกันเมื่อlayersมาก; วัดpairwisedistance/dirichletenergyและaccuracy1..8layers ไม่ดูเพียงtrainingloss
9. graphpoolmeansumตามgraphID; graphAfeatures[1,3]mean2,sum4; duplicateทุกnode meanคงเดิม,sumเพิ่ม2× แสดงขนาดgraphมีผลต่าง
10. linkscoreσ(huᵀhv); h0[1,0],h1[0,1]ได้.5; h0กับh0ได้.731059; negativesต้องไม่เป็นpositiveedgeหรือleakfutureedges
11. chainAhatข้อ2กับX=ones(3,1)ให้[.908248,1.149830,.908248]; symmetricnormalizationไม่รับประกันrow sum1
12. permutenodeIDsด้วยPแล้วGCN(PAPᵀ,PX)=PGCN(A,X); assertselfloopsdegree>0และisolatednodefiniteตามpolicy
13. เปรียบsum/meanaggregationโดยคงgraph/X/weightsและdegreeheterogeneity; reportnodeclassificationaccuracyพร้อมdegree-slice ไม่เทียบtensorเพียงscaleโดยไม่normalize
14. recordedgeindex(2,E),X(N,d),degrees(N),Ahat(N,N),logits(N,C),maskและmacroF1; graphbatchIDต้องไม่ปนedgeข้ามgraph
15. testedgeในtrainingadjacencyทำlinkpredictionleak; transductive/inductiveprotocolต่างกันต้องระบุและnegative samplingไม่ใช้futureknowledgeผิดpolicy
16. sparseGCNO(E d+Ndh),denseAhatmemoryO(N²); neighbor samplingจำกัดfanoutแต่เพิ่มstochasticvariance ต้องrecordlayers/fanout
17. เทียบdenseNumPyกับtorch.sparse.mmด้วยAhatweightsตรงและedgecoalescingpolicyเดียว; permuteIDstestช่วยจับaxisผิด
18. เปลี่ยนselfloopon/offคงdata/weights/seed; reportisolatednodesoutputและmacroF1หลายseed ไม่เหมารวมว่าทุกgraphต้องดีขึ้น
19. manifestเก็บnode/edgeIDmapping,graphsplit,edge-direction,selfloop,normalization,negative-sampling,seed,checkpointและmetricslices
20. ทดลอง“residualลดoversmoothingที่8layers”; baselineGCN8,เปลี่ยนresidualอย่างเดียว,วัดpairwisedistanceและinductiveholdoutmacroF1
## 38
1. rewards[1,2,3],γ.5ให้G0=1+.5·2+.25·3=2.75,G1=3.5,G2=3; γdiscountอนาคตไม่ลดimmediatereward
2. Vπ(s)=Σaπ(a|s)Σs'P(s'|s,a)[r+γVπ(s')]; V*(s)=maxaΣs'P[r+γV*(s')]; terminalcontinuation0
3. สองstates,s0actiongo→terminalr1,stay→s0r0,γ.9 เริ่มV0แล้วupdateได้V(s0)=1และคง1; iterateจนmaxabsdelta<tolerance
4. Q(s,a)←Q+α[r+γmaxQ(s',·)−Q]; Q2,r1,nextmax4,γ.5,α.1ให้Q2.1; terminaltargetrเท่านั้น
5. ε1สำรวจuniform,ε0greedy; decayเร็วเสี่ยงพลาดactionดี decayช้าสะสมregret; comparecumrewardและactioncoverageหลายseed
6. ringbufferเก็บ(s,a,r,snext,terminated)capacity3ใส่4รายการเหลือ3ล่าสุด; samplewithoutreplacementbatch≤sizeและไม่แก้transitionหลังstore
7. target=r+γ(1−terminated)maxQtarget(s'); r1,nextmax4,γ.5ให้3ถ้าnonterminalและ1ถ้าterminal; truncationไม่จำเป็นต้องterminalตามenvsemantics
8. REINFORCE gradientloss=−R·logπ(a); banditR1,π[a].25ให้loss1.386294; sampleactionแล้วoptimizerเพิ่มprobactionที่rewardเหนือbaseline
9. advantageA=G−Vφ(s),policyใช้stopgradient(A); valueloss(MSE(V,G))ฝึกbaselineแยกและไม่backproppolicylossเข้าGหรือbaselineโดยผิดตั้งใจ
10. A2,ratio1.3,ε.2 surrogate=min(2.6,2.4)=2.4; A−2,ratio.7ได้min(−1.4,−1.6)=−1.6; lossคือลบmean
11. Qtargetnonterminal3,current2,TDerror1,squaredloss1; terminaltarget1,TDerror−1; ข้อนี้ใช้แยกbootstrappingผิดที่terminal
12. assertterminaltargetไม่ขึ้นกับnextQ,transitionprobsต่อstateactionรวม1และvalueiterationtinyMDPได้V1; seedการsampleactionแยกenvseed
13. คงbandit/steps เปรียบε.1กับdecay1→.01; reportcumrewardและoptimalactionfractionเฉลี่ยหลายseedพร้อมinterval ไม่เลือกtrajectoryดีที่สุด
14. recordobs(B,d),action(B,)integer,Q(B,A),chosenQ(B,),rewards/dones(B,),episodereturnและTDerrorfinite
15. evalยังเปิดεสูงทำmetricปนexploration; train/testenvironmentseedsแยกและreportevaluationpolicyε; tunedrewardไม่แทนrealobjectiveอัตโนมัติ
16. tabularQO(SA)memory; DQNreplayO(capacity·obsdim),networkforwardO(BP)ตามlayers; envstepcostอาจdominantต้องวัดแยก
17. NumPyBellmanbackupเทียบexplicitenumerationบนtransitiontable; DQNtargettorch.no_gradต้องตรงtinyfixtureรวมterminalmask
18. targetnetworkupdateperiod10→100คงreplay/batch/seed; reportTDvarianceและreturnหลายseedโดยequalenvironmentinteractions
19. manifestเก็บenvversion/reward/terminationpolicy/γ/α/εschedule/seeds/replay/targetperiod/steps/checkpointและevaluationepisodes
20. ทดลอง“doubleDQNลดQoverestimationในnoisybandit”; baselineDQN,เปลี่ยนactionselectiontarget,วัดQbiasต่อtruevalueและreturnfixedseeds
## 39
1. logisticP(y|x)เป็นdiscriminative; autoregressiveP(x),VAE/GAN/diffusionสร้างsamplesเป็นgenerative; classifierfreeguidanceยังไม่ทำdiffusionเป็นclassifier
2. normalNLL=.5log(2πσ²)+(x−μ)²/(2σ²); x1,μ0,σ1ได้1.4189385; σต้อง>0และdensityอาจ>1จึงNLLติดลบได้บางvariance
3. P(a,b,c,d)=P(a)P(b|a)P(c|a,b)P(d|a,b,c); probabilities.5,.25,.5,.1ให้P.00625,NLL5.075174
4. ELBO=Eqlogpθ(x|z)−KL(qφ(z|x)||p(z)); maximizeELBOเทียบminreconstructionNLL+KLโดยreduction/likelihoodต้องตรง
5. GANเรียนsamplingtransformationโดยไม่มีtractableinverse/Jacobiandensityทั่วไป; discriminatorprobไม่ใช่pG(x)จึงใช้คำนวณlikelihoodตรงๆไม่ได้
6. energies[0,log2]ให้unnormalized[1,.5],Z1.5,prob[2/3,1/3]; finiteexampleคำนวณZได้แต่continuoushighDมักแพง
7. xt=√abar·x0+√(1−abar)ε; x02,ε−1,abar.25ได้1−.866025=.133975; ใช้εเดียวกันเมื่อเทียบabarเพื่อเห็นsignal/noise
8. real50%ที่−2และ50%ที่2,Gสร้างเฉพาะ2ทุกครั้งคือmodedroppingแม้sampleหนึ่งดูสมจริง; coverageหนึ่งในสองmode
9. qualityวัดความสมจริงต่อsampleส่วนcoverageวัดdistributionครบ; collapsedGอาจqualityสูงแต่coverageต่ำต้องวัดทั้งสองและunselectedsamples
10. texttableมีNLL,taskaccuracy,diversity,latency; imageมีdistributionmetric/precision-recall,coverage,conditionalconsistencyและsampleprotocol; ไม่เทียบFIDจากsamplesizeต่างกัน
11. x02,abar.25,ε−1ให้xt.133975และreconstruct(xt−√.75·ε)/.5=2; แยกsampleจากmean√abarx0=1
12. finiteenergyprobsรวม1; VAEKL0เมื่อposteriorเท่ากับprior; diffusionreconstructperfectepsilonกลับx0ด้วยallclose
13. compareGaussianσ.5/1บนfixedholdoutsamplesแล้วreportNLLแยกsampledispersion; σเล็กไม่แปลว่าดีเมื่อholdoutห่างmean
14. recordnoise(B,d),latent(B,k),outputs(B,d),NLL/ELBOส่วนประกอบและcoverage; ไม่เอาGANlossเทียบVAEELBOตรงๆ
15. cherry-picksamplesหรือใช้trainimagesเป็นFIDreferenceโดยไม่ระบุทำประเมินสูงเกิน; freezetestsamples/metric/samplecountก่อนเลือกmodel
16. autoregressivegenerationserialTsteps; diffusionTdenoisesteps; cost≈steps·networkforwardและmemoryactivationตามbatch ไม่กล่าวmodelfamilyหนึ่งเร็วเสมอ
17. compareGaussianNLLกับtorch.distributions.Normal.log_probและsum/meanreductionตรง; finiteenergyเทียบlogsumexpstability
18. เปลี่ยนsamplingsteps50→20ในdiffusionโดยmodel/seed/noiseเดียว; reportquality/coverage/latencyจริงหลายseedไม่อ้างfewerstepsเท่ากันทุกmodel
19. manifestเก็บtrain/referencehash/modelcheckpoint/noiseprior/sampler/steps/guidance/seed/samplesize/metricversion/hardware
20. ทดลอง“เพิ่มguidanceเพิ่มconditionalaccuracyแต่ลดcoverage”; baselineguidance1,เปลี่ยน1→3,วัดสองmetricพร้อมCIบนfixedpromptset
## 40
1. β=[.1,.2]ให้α=[.9,.8],abar=[.9,.72]; abarเป็นcumulativeproductไม่ใช่cumulativesumหรือ1−Σβ
2. recursiveforwardx1=√.9x0+√.1ε1,x2=√.8x1+√.2ε2; closedformdistributionmean√.72x0,var.28เท่ากัน แต่sampleไม่ตรงถ้าnoiseไม่ได้couple
3. signalcoeff√abarลดจาก1สู่0,noise√(1−abar)เพิ่มจาก0สู่1; SNR=abar/(1−abar),abar.25ให้1/3 ไม่ใช่noisevarianceเท่ากับβt
4. xhat0=(xt−√(1−abar)εpred)/√abar; perfectεทำxhat0=x0; abar0หารศูนย์ต้องexcludeหรือใช้parameterizationอื่น
5. μθ=1/√αt[xt−βt/√(1−abart)εθ]; predictionepsilonไม่ใช่x0ต้องใส่coefficientถูก; ancestralเพิ่ม√variance·zอีกที
6. posteriorvariance=βt(1−abarprev)/(1−abart); β2.2,abarprev.9,abart.72ได้.07142857; t1,abarprev1ให้0
7. seed7mixture2D,samplet/noise,trainMLPεpredด้วยMSEmean; logvalMSEแยกtและcomparezero-noisepredictorbaselineไม่อ้างภาพสวยแทนmetric
8. เริ่มxT~N(0,I),วนt=T..1คำนวณμและvarianceแล้วsample; t1ไม่เพิ่มnoiseตามposteriorvariance0; ใช้seedทำtrajectoryซ้ำได้
9. DDPMancestralมีnoiseแต่ละstep; deterministicDDIMη0ใช้x0hat/εpredและตารางabarข้ามเวลาโดยไม่มีfreshnoise; ไม่ใช่แค่ลบnoiseจากDDPMแล้วถือสูตรเดียวกัน
10. εguided=εuncond+s(εcond−εuncond); uncond.2,cond.5,s2ได้.8; s1เท่ากับconditional,s0unconditional; guidanceไม่updateweights
11. x02,abar.25,ε−1ให้xt.1339746และperfectreconstruct2; reversemeanต้องใช้β/αที่tเดียวกันไม่ใช้abarแทนα
12. testabarค่อยลดในschedulevalid0<β<1,posteriorvariance≥0,reconstructexactε; seedtrajectoryrepeatableและt0/t1boundaryไม่index−1ผิด
13. เปรียบlinear/cosinescheduleโดยsteps/modelbudgetเท่ากัน; recordSNRและvalnoiseMSEbyt พร้อมsamplequalityหลังtrainingจริง
14. recordx0/xt/εpred(B,2),t(B,),abart(B,1),finiteoutputsและvalMSE; broadcasttผิด(B,)อาจปนsampleกับfeature
15. เลือกsamplebestseedหรือใช้trainingpointsวัดcoverageอย่างเดียวทำคะแนนสูงปลอม; freezeheldoutdistribution/samplecount/evaluationnoise
16. trainingหนึ่งsampletใช้หนึ่งnetworkforwardต่อbatch; samplingTtimesnetworkcost; เก็บทุกtrajectoryเพิ่มO(TBd)memoryโดยไม่จำเป็นต่อsampling
17. NumPyforward/reversecoefficientsเทียบtorchfloat64กับschedule/noiseเหมือนกัน; statisticalforwardclosedformเทียบmean/varหลายdrawไม่expectsampleexactต่างnoise
18. guidance1→3คงmodel/noise/seeds/steps; reportconditionalhit-rateและmodediversityพร้อมlatencyของtwoforwardguidance
19. manifestเก็บβschedule/abarindexing/predictiontype/lossweighting/checkpoint/sampler/timesteps/η/guidance/noiseseed/evaluationversion
20. ทดลอง“DDIM20stepsคงcoverage≥95%ของDDPM100ใน2Dแต่เร็ว≥3×”; fixedmodel/noise/samplecountและcoverage/latencymeasured
## 41
1. normalizeeachrow e/‖e‖; [3,4]→[.6,.8]; zeroembeddingต้องerrorหรือpolicyชัดเจนไม่หาร0แล้วซ่อนNaN
2. imageI(B,d),textT(B,d),S=I@T.T/τ→(B,B); diagonalคือคู่positiveตามalignment ไม่ใช่everyhighsimilarityเป็นpositive
3. B3,S=I3logitsdiag1,offdiag0ให้CE=log(e+2)−1≈.551445ต่อrowและsymmetricmeanเท่ากัน; averageimage/textlossไม่sumถ้าใช้CLIPmean
4. τเล็กทำsoftmaxแหลมและgradientscale1/τสูง; τต้อง>0; logitsที่เปลี่ยนจากtemperatureไม่ใช่embeddingsretrained
5. Recall@Kคือfractionqueriesที่goldอยู่topK; queries3,hit2ให้2/3; multi-positiveต้องกำหนดanyhitหรือfractionpositivesก่อน
6. captionsซ้ำของภาพเดียวในbatchเป็นfalse negativesหากใช้onlydiagonal; ใช้multi-positivecontrastiveหรือdedupและreportcollisionpolicy
7. Qtext(B,Tt,d),K/Vimage(B,Ti,d)ให้A(B,h,Tt,Ti); maskเฉพาะvalidimagepatchesไม่causalข้ามmodality
8. latefusionผสมpooledvectorsราคาถูกแต่สูญเสียtokenalignment; crossattentionแพงกว่าจับregion-wordrelationได้; comparetoyspatialquestionaccuracy
9. visualfeatures(B,N,dv)@W(dv,dllm)→(B,N,dllm); biasbroadcastlastdimและprojectoroutputscaleตรวจตรงLLMembeddingrange
10. text-only/image-only/shuffledpairsfullmodelablation; shuffledimageควรลดtaskที่ต้องอ่านภาพ หากไม่ลดให้ตรวจshortcutleakageไม่อ้างmultimodalreasoning
11. I/Tidentity3,τ1ให้symmetricloss.551445; เปลี่ยนpairalignmentโดยไม่เปลี่ยนlabelsควรlossสูงขึ้นแม้scorematrixshapeเท่าเดิม
12. normrow≈1,similaritymatrixshape(B,B),permutebothmodalitiesเท่ากันไม่เปลี่ยนloss; permuteข้างเดียวต้องเปลี่ยนmatchinglabelsตาม
13. τ.1/.5คงpairedembeddingsแล้วreportentropy/loss; trainingrunต้องคงinit/split/budgetและmeasureRecall@Kจริง
14. recordimagefeatures/textfeatures/projectedtokens/similaritymatrix/positiveIDsและRecall@1/5; B1contrastiveloss0ไม่พิสูจน์alignment
15. ภาพเดียวหลายcaptionข้ามsplitทำretrievalleak; splitตามimage/sourcegroupและdedupทั้งmodalities ไม่ให้filenameบอกlabel
16. similaritiesO(B²d)memoryO(B²); crossattentionO(BTtTi d),visualprojectorO(BNdvdllm); encodercostต้องวัดรวมด้วย
17. NumPyrow-normalize+logsumexpCEเทียบtorchfunctional.normalize+cross_entropyสองทิศmean; epsและpositivealignmentตรง
18. เปลี่ยนprojectorlinear→2layerMLPคงencoders/data/seed; reportparameters/Recall@K/latencyและหลายrunsไม่เทียบเฉพาะtrainloss
19. manifestเก็บpairIDs/modalityhash/encoder/tokenizer/imagepreprocess/projector/temperature/false-negativepolicy/split/seed/metrics
20. ทดลอง“crossattentionช่วยspatialrelationtask≥5จุดเทียบlatefusion”; fixedencoder/data/budget,holdoutrelationsและlatency/memorybudget
## 42
1. duration=N/sample_rate; 16000samplesที่16kHz=1วินาที,48000samplesที่48kHz=1วินาที; channelsไม่คูณduration
2. Nyquist=sample_rate/2; 8kHz→4kHz,16kHz→8kHz,48kHz→24kHz; downsampleต้องlowpassก่อนป้องกันaliasing
3. frameL4,hop2ของ[0,1,2,3,4,5]ได้[[0,1,2,3],[2,3,4,5]]แบบnopadding; count1+floor((N−L)/hop)
4. rectangularตัดขอบทันทีเกิดspectralleakageมากเมื่อsineไม่ลงFFTbin; Hannลดsidelobesแต่mainlobeกว้างและamplitudecorrectionต่าง
5. fs8000,NFFT800,sine1000Hzมีpeakbin100เมื่อใช้alignedframe; `abs(rfft(frame*window))` ให้frequencyk·fs/NFFT
6. mel=2595log10(1+f/700),เลือกmel-spacededgesแล้วสร้างtriangularweightsในFFTfrequencybins; filtersแต่ละตัวไม่จำเป็นrow-sum1หากไม่normalize
7. logmel=log(max(M@abs(STFT)**2,eps)); powerไม่ใช่magnitudeและeps/logbaseต้องประกาศ; zero-waveformต้องfinitefloor
8. CTCcollapseซ้ำก่อนremoveblank: [a,a,blank,a,b,b]→aab; [a,blank,a]→aa ส่วน[a,a]→a
9. reference“I like AI”,hyp“I love AI”มีS1,D0,I0,N3ให้WER1/3; CERหน่วยcharacterและnormalizationต้องversioned โดยไทยไม่ใช้whitespacewordassumptionเงียบๆ
10. seed7synthetictonesA/B,splitwaveformsก่อนframing; encoderframewise→CTClogits,blankIDชัด; targetlength≤inputlengthและmeasuresequenceerrorจริง
11. framecountN16000,L400,hop160,nopadได้98; duration1secแต่framecoverageสิ้นสุด15920samples ต้องระบุcenterpaddingถ้าlibraryได้101frames
12. CTCblankinsertionระหว่างidenticallabelsเปลี่ยนdecodedlengthถูก,STFTsinepeakตรงbinและzeroaudiofeaturesfinite; noframesเมื่อN<Lต้องpolicy
13. เปรียบHann/rectangularด้วยsine1003Hzfs8000คงNFFT; วัดsidelobeenergy/peakwidthไม่เลือกภาพspectrumที่ขยายaxisต่างกัน
14. recordwaveform(B,N),frames(B,T,L),STFT(B,T,F),mel(B,T,M),CTClogits(T,B,C),lengthsและCER/WER
15. speakerเดียวกันทั้งtrain/testหรือsegmentoverlapทำASRดูดีเกิน; splitbyspeaker/sessionก่อนchunkและreportnoiseslices
16. STFTO(TNFFTlogNFFT),melO(TFM),CTCdynamicprogramO(TU); streaminglatencyรวมframe/contextlookaheadไม่ใช่เฉพาะnetworktime
17. np.fft.rfftเทียบtorch.stftโดยcenter/window/hop/normalizationตรง; CTCgreedyเทียบmanualcollapsefixtureไม่เทียบbeamdecoderที่ใช้LM
18. เปลี่ยนhop160→320คงfs/window/modelbudget; reportframecount/latency/CERและเวลาresolution ไม่อ้างลดframesแล้วqualityคงเสมอ
19. manifestเก็บsample-rate/channelpolicy/resampler/window/hop/NFFT/mels/logfloor/blankID/tokenizer/speakersplit/seed/checkpoint
20. ทดลอง“Hannลดtoneclassificationerrorภายใต้off-binfrequency”; rectangularbaseline,fixedtoneSNR/splitและCER/latencymeasuredหลายseed
## 43
1. logs(u0,i1),(u0,i2),(u1,i2)ให้matrix[[0,1,1],[0,0,1]]เมื่อbinaryimplicit; repeatcounts/ratingsต้องแยกความหมายจากmissing
2. countitemsแล้วsortcountdesc,tieitemID; user0consumed1,2ต้องexcludeทั้งคู่,เลือกitem0แม้popularน้อย ไม่แนะนำitemที่ผู้ใช้มีแล้ว
3. cosine[1,1]กับ[0,1]=1/√2; sparsezerosไม่ใช่negativepreferenceเสมอ; similaritynormalizeหลังจัดIDmappingตรงกัน
4. predictionpᵤᵀqᵢ,error=pred−rating; updatep←p−η(2e q+2λp),qใช้pเก่าก่อนupdate; p1,q2,r3,e−1,η.1,λ0ให้p1.4,q2.2
5. pred=μ+bu+bi+pᵤᵀqᵢแยกglobal/user/itembias; regularizeแต่ละส่วนตามนิยามและinitialbias0ทำinitialpredμเมื่อfactors0
6. BPRsample(u,i+,j−)ที่jไม่เคยconsumed; loss=softplus(−(sui−suj)); margin2ให้loss.126928และgradientผลักpositiveเหนือnegative
7. relevant{a,c},top3[a,b,c]ได้P@3=2/3,R@3=1,DCG=1+1/log2(4)=1.5,IDCG=1+1/log2(3),NDCG≈.919721
8. randomsplitอาจให้อนาคตเข้าฝึกและประเมินeasyitems; temporaltrainpast/testfutureตรงproductionมากกว่าแต่ต้องจัดcoldstartและglobalcutoffชัด
9. usertowerf(u)และitemtowerg(i)ให้vectorsdเดียว,score=f·g; retrieveprecomputeditemvectors,trainingnegativepolicyต้องตรงcatalogavailability
10. newuserใช้context/popularityตามregionที่มีสิทธิ์, newitemใช้contentembedding; reportcold-user/cold-itemslicesและavoidแนะนำunavailableitem
11. factorsp[1,0],qa[0,1],qb[1,1]ให้scores0,1; normalizationเปลี่ยนdotเป็นcosineจึงอาจเปลี่ยนrankingเมื่อitemnormต่าง
12. recommendationsไม่ซ้ำ/ไม่consumed,rankingstabletiesและno-testinteractiontrain; BPRpositiveเหนือnegativeแล้วlossควรลดเมื่อเพิ่มmargin
13. popularityเทียบMFคงtemporalcutoff/candidatecatalog; reportRecall@10/NDCG@10และcoverageแยกcoldstart ไม่เปลี่ยนnegativepoolระหว่างmodels
14. recordusers(B,),items(B,),factors(B,d),scores(B,K),candidatevalidmaskและrankingmetricsตามfull/sampledcatalogprotocol
15. sampledeasynegativesทำmetricสูงปลอมเทียบfullcatalog; reportcandidatepool/negativepolicyและไม่ใช้futureinteractionเป็นtrainnegativeโดยไม่พิจารณา
16. MFscoreallcatalogO(Nitems d)ต่อuser,ANNลดsearchแลกrecall; trainingBPR O(Bd)แต่embeddingtablesO((U+I)d)
17. NumPyMFforward/gradเทียบtorchEmbeddingdotMSEด้วยreductionและλconventionเดียว; rankingsเทียบexactsearchก่อนANN
18. เปลี่ยนnegativeuniform→popularity^.75คงdata/seed/updates; reportNDCG/longtailcoverageกับtraincostหลายseed
19. manifestเก็บcutoff/user-itemmapping/catalogsnapshot/consumedpolicy/negative-sampling/factordim/seed/metricsandidateprotocol/checkpoint
20. ทดลอง“contentfallbackเพิ่มcolditemRecall@10โดยwarmuserNDCGลด≤.01”; popularitybaseline,fixedtemporalholdoutและsliceCI
## 44
1. xyxy[1,2,5,8]→xywh[1,2,4,6]→cxcywh[3,5,4,6]; continuouscoordinatesไม่มี+1pixelเว้นประกาศinclusiveintegerpolicy
2. A[0,0,2,2],B[1,1,3,3]intersection1,area4+4,union7ให้IoU1/7≈.142857; disjoint0และidentical1
3. broadcastboxesA(N,1,4),B(1,M,4),intersectionw/hclip≥0→IoU(N,M); zeroareaboxต้องrejectหรือdefine0ไม่NaN
4. sortscoredesc,tieoriginalID; เลือกboxแรกแล้วsuppressIoU>thresholdเฉพาะsameclass; classต่างกันoverlap1ก็ยังเก็บ
5. thresholdสูงsuppressน้อยจึงpredictionsมากขึ้น แต่อาจfalsepositivesเพิ่ม; reportprecision/recallไม่เลือกจากtestset
6. score-descpredictionจับGTsameclassIoUสูงสุดที่ยังunmatchedและ≥threshold; GTหนึ่งตัวรับTPได้ครั้งเดียว,duplicatepredictionเป็นFP
7. TP2,FP1,FN2ให้precision2/3,recall.5; ต้องระบุIoUthresholdและscorethresholdก่อนนับ ไม่เอาค่าIoUmeanแทนAP
8. masksA={pixels1,2},B={2,3}ได้intersection1,union3,IoU1/3,Dice2/(2+2)=.5; bothemptypolicyต้องระบุ
9. seed7syntheticcircles/rectangles,image32²; splitก่อนaugmentation,trainUNetBCElogitsแล้วmeasurepixelIoU/Diceกับempty-maskbaselineและtestholdoutshapes
10. semanticlabelsให้classต่อpixelเช่นcar1ทุกคัน; instancelabelsให้IDแยกแต่ละคันแม้classเดียว; panopticรวมclassและinstance
11. tinyboxesข้อ2IoU1/7; score.9/.8sameclass,NMSthreshold.1เก็บเพียงแรก แต่threshold.2เก็บทั้งสอง ต้องไม่NMSข้ามclass
12. identicalboxesIoU1,symmetryIoU(A,B)=IoU(B,A),bounds[0,1]; class-awareNMSเก็บsameboxdifferentclassและduplicateGTmatchไม่เกิดสองTP
13. เปรียบNMSthreshold.3/.7โดยpredictions/splitเดียว; reportTP/FP/FN,Recallและlatencyก่อนเลือกvalidationthreshold
14. recordboxes(N,4),scores(N,),classes(N,),IoUmatrix(N,M),matchedIDsและprecision/recall; segmentationlogits(B,1,H,W)ตรงmask
15. cropsofimageเดียวข้ามsplitหรือเลือกscore/IoUthresholdจากtestทำleak; splitbyimage/patientและreportobject-size-slices
16. pairwiseIoUO(NM),naiveNMSO(N²),maskmetricsO(HW); detectorbackbonecostแยกจากpostprocessingและpeakmemoryของIoUmatrix
17. NumPyboxIoU/NMSเทียบtorchvisionเมื่อoptionalbinaryเข้ากันได้; coreIoUoraclemanualต้องผ่านเสมอและtiepolicyอาจทำindicesต่าง
18. เปลี่ยนBCE→BCE+DicelossคงUNet/data/initbudget; reportsmallobjectDiceกับfalsepositiveareaหลายseed ไม่เทียบlossscaleตรงๆ
19. manifestเก็บcoordinateconvention/resize-letterboxtransform/classIDs/score-NMS-IoUthreshold/split/hash/seed/checkpoint/metricversion
20. ทดลอง“Diceadditionเพิ่มsmallshapeIoU≥.03เทียบBCE”; holdoutsize-shift,fixedtrainingbudget,reportCIและlatency/memory
