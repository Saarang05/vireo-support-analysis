import pandas as pd, glob, numpy as np
U='/mnt/user-data/uploads/'
t=pd.read_csv(glob.glob(U+'*tickets.csv')[0],parse_dates=['created_at','first_response_at','resolved_at'])
o=pd.read_csv(glob.glob(U+'*orders.csv')[0],parse_dates=['order_date']);p=pd.read_csv(glob.glob(U+'*products.csv')[0])
a=t[t.order_id.notna()].merge(o[['order_id','lot_code']],on='order_id',how='left')
b=t[t.order_id.isna()].sort_values('created_at')
b=pd.merge_asof(b,o.sort_values('order_date').rename(columns={'sku':'product_sku'})[['customer_id','product_sku','order_date','lot_code']],left_on='created_at',right_on='order_date',by=['customer_id','product_sku'])
m=pd.concat([a,b])
bad=m.product_sku.eq('VA-EB-PL2')&m.lot_code.str.match(r'PL2-25(10|11)-')
m['bad']=bad
print('bad-lot tickets',bad.sum(),'of',len(m))
# rate comparison for PL2 other lots
pl=m[m.product_sku=='VA-EB-PL2']
og=o[o.sku=='VA-EB-PL2'].groupby('lot_code').size()
print(og.sort_index())
for flag,g in pl.groupby('bad'):
    print(flag,len(g),'rep',(g.replacement_issued=='Y').sum(),'csat',g.csat_score.mean().round(2))
base_rate=(pl[~pl.bad].replacement_issued=='Y').sum()/o[(o.sku=='VA-EB-PL2')&~o.lot_code.str.match(r'PL2-25(10|11)-')].shape[0]
badorders=o[(o.sku=='VA-EB-PL2')&o.lot_code.str.match(r'PL2-25(10|11)-')]
print('bad orders',len(badorders),'units',badorders.qty.sum(),'base rep/order',round(base_rate,3))
rep_bad=((m.bad)&(m.replacement_issued=='Y')).sum()
exp=base_rate*len(badorders);print('replacements bad lots',rep_bad,'expected',round(exp,1),'excess',round(rep_bad-exp,1))
print('excess tickets', bad.sum()-(pl[~pl.bad].shape[0]/o[(o.sku=='VA-EB-PL2')&~o.lot_code.str.match(r'PL2-25(10|11)-')].shape[0])*len(badorders))
# agent view
m['rep']=m.replacement_issued=='Y'
ag=m[m.csat_score.notna()].groupby('agent_id').agg(n=('csat_score','size'),csat=('csat_score','mean'),badshare=('bad','mean')).sort_values('csat')
print(ag.head(12).round(2));print(np.corrcoef(ag.csat,ag.badshare)[0,1])
