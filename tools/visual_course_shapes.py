"""Authored operation/state assessments, never dimensions of a presentation table."""
def matrix(v, label, left='X', right='W'):
 n,d,h=(2+v,3+v,4+v)
 return (f'{label}: {left} shape ({n},{d}), {right} shape ({d},{h})',f'{left}@{right} ให้ shape ใด?',f'({n},{h})',[f'({h},{n})',f'({n},{d},{h})'],f'แกนด้านใน {d} ตรงกันแล้วถูก sum เหลือแกนตัวอย่าง {n} และ output {h}')
def attention(v,label):
 tq,tk,dk=2+v,4+v,3+v
 return(f'{label}: Q({tq},{dk}),K({tk},{dk}),V({tk},2)','Q@K.T มี shape ใด?',f'({tq},{tk})',[f'({tk},{tq})',f'({dk},{dk})'],f'หนึ่งแถวต่อ query ({tq}) หนึ่งคอลัมน์ต่อ key ({tk}); แกน feature {dk} ถูก dot product')
def cnn(v):
 h,k,p,s=4+v,2,0,2
 size=(h+2*p-k)//s+1
 return(f'CNN input {h}×{h},kernel{k}×{k},padding{p},stride{s}', 'output ช่องเดียวมี shape ใด?',f'({size},{size})',[f'({h},{h})',f'({size+1},{size+1})'],f'floor(({h}+2×{p}−{k})/{s})+1={size} ต่อแกน; valid flat index0–{size*size-1}')
def state(context,prompt,answer,wrong,reason):
 text=lambda x:format(x,'.8g')if isinstance(x,(int,float))else str(x)
 return(context,prompt,text(answer),list(map(text,wrong)),reason)
# Numbers are intentionally small and independent of a learner's edited workbench state.
def case(n,v):
 b,t,d,h=2+v,3+v,4+v,5+v
 mat={2:'matrix multiplication',3:'NumPy broadcast หลัง linear projection',18:'Dense layer',23:'Keras Dense',30:'Transformer FFN ขาแรก',55:'classification head',74:'visual-token projector'}
 if n in mat:return matrix(v,mat[n], 'X.T'if n==19 else'X','dZ'if n==19 else'W')
 if n in [29,36]:return attention(v,'self-attention'if n==29 else'cross-attention; source/target lengths ต่างกัน')
 if n==24:return cnn(v)
 if n==1:return state(f'values=[1,{2+v},3],เริ่มtotal=0; loopบวกทุกสมาชิก','หลัง loop total เป็นเท่าไร?',6+v,[5+v,3],'accumulator เปลี่ยนตามสมาชิกแต่ละตัว ไม่ใช่นับจำนวนรอบ')
 if n==4:return state(f'train X({b},{d}),imputer/scaler ไม่ลบแถวหรือเพิ่มfeature','หลัง transform มี shape ใด?',f'({b},{d})',[f'({d},{b})',f'({b-1},{d})'],'fit statistics ไม่ได้เปลี่ยนจำนวน samples/features ภายใต้ operation ที่กำหนด')
 if n in [5,6,7,13]:return state(f'X({b},{d}), w({d},), b scalar; prediction=X@w+b','prediction ของ batch มี shape ใด?',f'({b},)',[f'({b},{b})',f'({d},)'],'dot ลดแกน feature เหลือหนึ่ง prediction ต่อ sample; target ต้อง shape เดียวกันก่อน loss')
 if n in [8,14,45]:return state(f'X samples({b},{d}), prototypes/candidates({h},{d})','pairwise distances/similarities มี shape ใด?',f'({b},{h})',[f'({h},{b})',f'({b},{d},{h})'],'หนึ่ง score ต่อคู่ sample กับ candidate; feature dimension ถูก reduce')
 if n==9:return state(f'Gaussian NB X({b},{d}),C={h} classes; sum log likelihood over features','class log-joint ต่อ sample มี shape ใด?',f'({b},{h})',[f'({b},{d})',f'({d},{h})'],'ลดแกน feature แต่เก็บแกน sample และ class ก่อน normalize posterior')
 if n==10:return state(f'node มี{8+v}samples; splitส่งซ้าย{3+v},ไม่ทิ้งแถว','ฝั่งขวามีกี่ samples?',5,[4,8+v],'left+right=parent; equality-at-threshold policyต้องส่งแต่ละแถวไปด้านเดียว')
 if n==11:return state(f'forest{3+v}trees,prob tensor({3+v},{b},{h})=trees×samples×classes','mean(axis=0) มี shape ใด?',f'({b},{h})',[f'({3+v},{h})',f'({b},)'],'เฉลี่ยต้นไม้แล้วเก็บ probability ทุก class ของทุก sample')
 if n==12:return state(f'squared-error boosting y({b},),prediction({b},)','negative gradient y−prediction มี shape ใด?',f'({b},)',[f'({b},{b})',f'({d},)'],'weak learner รับ residual ต่อ sample ไม่ใช่ feature matrix ที่ broadcast ใหม่')
 if n==15:return state(f'PCA X({b},{d}),เก็บk={1+v} components shape({1+v},{d})','(X−mean)@components.T มี shape ใด?',f'({b},{1+v})',[f'({1+v},{b})',f'({b},{d})'],'เก็บk≤min(n,d)และprojectด้วยtranspose; ไม่ขอPCAcomponentsมากกว่าrankที่มี')

 if n==16:return state(f'X({b},{d}),detector คืน anomaly score หนึ่งค่าต่อ sample','scores มี shape ใด?',f'({b},)',[f'({d},)',f'({b},{d})'],'feature axes ใช้คำนวณscoreแล้ว reduce เหลือ sample axis')
 if n==17:return state(f'series{10+v}จุด,lags{2+v},ไม่padding','จำนวน supervised rows เมื่อใช้ทุก lagครบเท่าไร?',8,[10+v,7],'ตัดช่วงแรก L จุด จึง n−L rows; feature_timeต้องก่อนtarget_time')
 if n==19:return state(f'X({b},{d}),dZ({b},{h})','dW=X.T@dZ มี shape ใด?',f'({d},{h})',[f'({h},{d})',f'({b},{h})'],f'X.T({d},{b}) คูณ dZ ลดbatch axis เหลือinput×outputfeatures')
 if n==20:return state(f'SGD currentθ={2+v},lossθ²,η=.1,updates0','กดหนึ่งก้าว current θ เป็นเท่าไร?',.8*(2+v),[2+v,.9*(2+v)],'g=2θ, θnew=θ−.1·2θ=.8θ; เปิดหน้า/แก้ηไม่ถือเป็นupdate')
 if n==21:return state(f'logits({b},{t},{d}); softmax class axis−1','sum(prob,axis=−1) มี shape ใด?',f'({b},{t})',[f'({b},{d})',f'({d},)'],'ลด class axis; แต่ละ batch/token ต้องได้ probability mass1')
 if n==22:return state(f'PyTorch Linear: X({b},{d}),weight({h},{d}),bias({h},)','X@weight.T+bias มี shape ใด?',f'({b},{h})',[f'({h},{b})',f'({b},{d})'],'torch Linear เก็บ weight=(out_features,in_features)จึงต้องtransposeเมื่อเขียนmatmulเอง')

 if n in [25,26]:return state(f'RNN/LSTM layers{h},batch{b},hidden{d},unidirectional','final hidden state มี shape ใด?',f'({h},{b},{d})',[f'({b},{t},{d})',f'({b},{d})'],'layer axis ไม่ใช่ time axis; LSTM cell stateมี shapeเดียวกับhidden')
 if n in [27,39]:return state(f'VAE batch{b},latent{h}; μ/logvar({b},{h})','ε และ sampled z ต้อง shape ใด?',f'({b},{h})',[f'({h},)',f'({b},{d})'],'reparameterization ใช้ noise ต่อ sample ต่อ latent coordinate ไม่แชร์sampleโดยเงียบ')
 if n==28:return state(f'D logits({b},1),binary labels สำหรับทุก sample','labels เพื่อ BCE ควร shape ใด?',f'({b},1)',[f'({b},)',f'({b},{b})'],'ones_like(logits) ป้องกัน target broadcast เป็น pairwise matrix')
 if n in [31,74]:return state(f'image{8+4*v}×{8+4*v},patch2×2,noCLS','มี patch tokens กี่ตัว?',((8+4*v)//2)**2,[((8+4*v)//2),((8+4*v)//2)**2+1],'patch count=(H/P)(W/P); CLS ถ้าใช้ค่อยเพิ่ม1')
 if n==32:
  tokens=['a','b','a','c']if v==1 else ['a','b']*(2 if v==0 else 3);count=sum(tokens[i:i+2]==['a','b']for i in range(len(tokens)-1))
  return state(f'tokens={tokens}','bigram(a,b) ปรากฏกี่ครั้ง?',count,[count+1,0],'นับ adjacent pairs ตามตำแหน่งจริง ไม่เชื่อม end กลับ start')

 if n==33:return state(f'embedding table(V,{d}),lookup IDs({b},{t})','lookup output มี shape ใด?',f'({b},{t},{d})',[f'({b},{d})',f'({t},{d})'],'lookupคงทุกindexaxisแล้วเพิ่มembeddingfeatureaxisท้ายสุด')
 if n==34:return state(f'BERT hidden({b},{t},{d}),token-head W({d},{h})','token-classification logits มี shape ใด?',f'({b},{t},{h})',[f'({b},{h})',f'({t},{d})'],'linear headใช้sharedweightsต่อtokenและเก็บsequenceaxis')
 if n in [35,48,62]:return state(f'KVcache({b},{h},{t},{d})=batch×kvheads×past×head_dim; append1token','cache หลัง append มี shape ใด?',f'({b},{h},{t+1},{d})',[f'({b},{h+1},{t},{d})',f'({b},{h},1,{d})'],'appendตามsequenceaxisเท่านั้น,K/Vเก่าคงเดิมและqueryไม่ถูกappendเข้าcache')
 if n==37:return state(f'adjacency({t},{t}),node features({t},{d}),W({d},{h})','Ahat@X@W มี shape ใด?',f'({t},{h})',[f'({h},{t})',f'({t},{t})'],'aggregateเก็บnodeaxisและprojectionเปลี่ยนfeaturewidth')
 if n==38:return state(f'Qlearning r={1+v},γ=.9,nextmax10,terminated=True','TD target เท่าไร?',1+v,[10+v,11+v],'terminalปิดbootstrap จึง target=r ไม่ใช้nextQ')
 if n==40:return state(f'x0({b},{d}),abar per sample({b},1)','epsilon สำหรับ noising ต้อง shape ใด?',f'({b},{d})',[f'({b},)',f'({d},)'],'noiseเป็นอิสระทุกsample-feature; abar broadcastตามfeatureได้แต่εต้องไม่แชร์ข้ามsample')
 if n==41:return state(f'pairedimage/text embeddings({b},{d})ทั้งคู่','I@T.T contrastive logits มี shape ใด?',f'({b},{b})',[f'({d},{d})',f'({b},{d})'],'ทุกภาพเทียบทุกข้อความในbatch; diagonalตามpairIDsเป็นpositive')
 if n==42:return state(f'waveform N={16+4*v},frame4,hop2,nopadding','จำนวน frames เท่าไร?',7+2*v,[8+2*v,4+v],'1+floor((N−frame)/hop) โดยไม่centerpadding')
 if n==43:return state(f'user factors({b},{d}),item factors({h},{d})','user@item.T scores มี shape ใด?',f'({b},{h})',[f'({h},{b})',f'({b},{d})'],'หนึ่งscoreต่อuser-itemcandidatepair ไม่ใช่outputembedding')
 if n==44:return state(f'predicted boxes({b},4),groundtruth({h},4)','pairwise IoU มี shape ใด?',f'({b},{h})',[f'({h},{b})',f'({b},4)'],'แต่ละcellคือIoUของคู่pred/GTก่อนone-to-onematching')
 if n in [46,72]:return state(f'context budget{10+v},segmentslen[6,5,4],greedyตามrankและskipชิ้นที่เกิน','ไล่ตามrankและข้ามชิ้นที่ใส่ไม่พอ ใช้ tokens รวมเท่าไร?',10 if v==0 else 11,[9,15],'budget10เลือก6+4; budget11/12เลือก6+5; ไม่truncateโดยไม่ประกาศ')
 if n==47:return state(f'max_steps={3+v},completed_steps={3+v},nexttoolเสนอมาอีก1','executorควรอยู่stateใด?','stopped:step_limit',['executing','reset_budget'],'ตรวจbudgetก่อนexecute; modelproposalไม่เพิ่มbudgetหรืออนุญาตsideeffectเอง')
 if n==49:return state(f'UTF8 basebytes0..255,reservedspecials={2+v},mergesยังไม่เริ่ม','new merge ID แรกเท่าไร?',258+v,[256,257+v],'ต้องข้ามreservedIDsทั้งหมดจึงไม่ชนBOS/EOS/specials')
 if n in [50,80]:return state(f'token window length{t+1},ใช้shiftหนึ่งตำแหน่ง','input/target แต่ละชุดยาวเท่าไร?',t,[t+1,t-1],'x=tokens[:-1],y=tokens[1:]จึงยาวwindow−1และtargetเป็นnexttoken')
 if n==51:return state(f'docA{2+v}tokens,EOS1,docB2tokens,concatenateall','packed stream ก่อนตัดblockยาวเท่าไร?',5+v,[4+v,6+v],'นับEOSที่คั่นจริงหนึ่งตัวตามโจทย์; EOSท้ายdocBยังไม่ถูกเพิ่มในcaseนี้')
 if n==52:return state(f'microbatch{b},accum{h},worldsize2,ทุกrankbatchเท่ากัน','global batch เท่าไร?',b*h*2,[b*h,b*2],'globalbatch=microbatch×accumulation×worldsize; tokenbatchยังต้องคูณvalidcontext')
 if n==53:
  count=[1023,1024,1025][v];grid=(count+255)//256
  return state(f'N={count},block256','grid blocks ขั้นต่ำเท่าไร?',grid,[grid-1,grid+1],f'ceil({count}/256)={grid}; Nหารblockลงตัวไม่ต้องเพิ่มอีกblock และthreadsที่เกินNต้องboundcheck')

 if n==54:return state(f'gradientnorm{4+v},lossscale8,clipmax2','หลังunscaleแล้วclip gradientnormเท่าไร?',2,[.25,16],'unscaleคืนtruegradientก่อนclip; ไม่clipscalednormแล้วหาร8')
 if n==56:return state(f'W({d},{d}),LoRA A({2+v},{d}),B({d},{2+v})','B@A มี shape ใด?',f'({d},{d})',[f'({2+v},{2+v})',f'({d},{2+v})'],'inner rankแกนhถูกsum จึงΔWตรงbaseWshape')
 if n==57:return state(f'IDs=[10,20,{2+v}],role=[user,assistant,assistant]; shift1','valid supervised targets หลังshiftมีกี่ตัว?',2,[1,3],'target[20,EOS]เป็นassistantทั้งคู่; ไม่รวมuserID10')
 if n==58:return state(f'PPO oldprob.2,newprob{.3+.02*v}','probability ratio new/old เท่าไร?',(.3+.02*v)/.2,[.2/(.3+.02*v),.1+.02*v],'ratio=exp(lognew−logold)=new/old ไม่ใช่ผลต่างprobability')
 if n==59:return state(f'tokenlogps=[−.1,−.2,−.{3+v}],completionmask=[0,1,1]','summed completion log-prob เท่าไร?',-.2-(3+v)/10,[-.1,-.1-.2-(3+v)/10],'ใช้เฉพาะcompletionpositionsแล้วsum; promptpositionแรกไม่อยู่objective')
 if n==60:return state(f'N={5+v}examples,bootstrapหนึ่งreplicateแบบIIDwithreplacement','replicate ต้อง draw กี่ indices?',5+v,[4+v,1],'bootstrapแต่ละreplicateมีขนาดnเท่าoriginal แต่indicesซ้ำ/ขาดบางตัวได้')
 if n==61:return state(f'weights{16+8*v}ตัว,group size8','ต้องเก็บscaleกี่กลุ่ม?',2+v,[1,8],'ceil(P/group_size); metadataเพิ่มตามgroupcount ไม่ใช่ตามจำนวนbitsอย่างเดียว')
 if n in [63,67]:return state(f'workers2,service_rate{5+v}/sต่อworker,arrival{12+2*v}/s','service capacity รวม requests/s เท่าไร?',10+2*v,[12+2*v,5+v],'capacity=workers×service_rate; arrivalมากกว่าcapacityทำqueueโต ไม่ใช่เพิ่มthroughputเอง')
 if n==64:return state(f'desired replicas{4+v},maxSurge=1pod,oldpodsยังอยู่','จำนวนpodสูงสุดระหว่างrollingupdateเท่าไร?',5+v,[4+v,3+v],'surgeเพิ่มจากdesiredหนึ่งpod ต้องมีresourceสำหรับpeakนี้จริง')
 if n==65:return state(f'quality{.82+.01*v},gate≥.8;p95=210ms,gate≤200ms','promotion stateควรเป็นอะไร?','reject:latency',['promote','unknown:quality'],'requiredgatesใช้AND; latencyfailแม้qualityผ่าน ห้ามเฉลี่ยกลบ')
 if n==66:return state(f'max_event_time12:00,allowed_lateness={5+v}min','watermark อยู่ที่นาทีใดก่อน12:00?',5+v,[0,10+v],'watermark=maxeventtime−allowedlateness ไม่ใช่processingtimeล่าสุด')
 if n==68:return state(f'toolbudget{3+v},used{2+v},nextrequestต้อง2calls','admissionstateควรเป็นอะไร?','reject:budget',['execute','increase_budget'],'remaining1น้อยกว่าrequested2 ต้องปฏิเสธก่อนsideeffect')
 if n==69:return state(f'linear f(x)={2+v}x,baseline0,x3','IG attribution เท่าไร?',3*(2+v),[2+v,4+v],'IG=(x−baseline)·constantderivativeและรวมเท่ากับoutputdifference')
 if n==70:return state(f'weights{8+4*v}ตัว,zero{4+2*v}ตัว','sparsityเท่าไร?',.5,[.25,1],'zero_count/total_count=.5; ไม่ใช่จำนวนbytesที่ประหยัดจริง')
 if n==71:return state(f'tokens{8+4*v},topk2,experts4,capacityfactor1','capacityต่อexpertตามceil(CF·N·k/E)เท่าไร?',4+2*v,[2+v,8+4*v],'นับassignmentsN·kก่อนหารexpertsและปัดขึ้น; overflowpolicyแยกต่างหาก')
 if n==73:return state(f'tokenbudget{1000+200*v},candidates4,ยังไม่หักverifier','แบ่งเท่ากัน candidateละกี่tokens?',250+50*v,[1000+200*v,500+100*v],'หารtotalbudgetด้วยN; actualbenchmarkต้องรวมverifieroverheadด้วย')
 if n==75:return state(f'audio{2+v}seconds,sample_rate16000Hz,mono','waveformมีsamplesกี่ตัว?',16000*(2+v),[16000,1000*(2+v)],'samples=seconds×samples/second; channelsกำหนดstorageอีกแกน')
 if n==76:return state(f'linear dynamics xnext=.8x+.2u,x={1+v},u=−1','xnextเท่าไร?',.8*(1+v)-.2,[.8*(1+v)+.2,1.2+v],'modelอ่านcurrentstate/actionแล้วทำtransitionครั้งเดียว ไม่ใช่แอบrolloutหลายก้าว')
 if n==77:return state(f'actionlimit[−1,1],proposedaction{2+v}','หลังclipได้actionเท่าไร?',1,[2+v,-1],'clipรักษาขอบบน1ก่อนส่งsimulator; nonfiniteactionsต้องreject')
 if n==78:return state(f'INT8tensor{1000+100*v}elements,ไม่รวมscale/metadata','raw storageกี่bytes?',1000+100*v,[4000+400*v,2000+200*v],'INT8ใช้1byte/element; peakRAMยังรวมactivation/arena/buffers')
 if n==79:return state(f'runconfig A={{a:{1+v},b:2}},B={{b:2,a:{1+v}}},canonicalJSONsortkeys','fingerprintsควรสัมพันธ์อย่างไร?','equal',['different','depends_on_insertion_order'],'canonicalkeyorderingทำให้configความหมายเดียวกันserializebytesเหมือนกัน')
 raise ValueError('Missing authored state/shape case for '+str(n))
