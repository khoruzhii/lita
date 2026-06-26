#------------------------------------------------------------------------------
# -*-mapleV-*-
#------------------------------------------------------------------------------
# Odd-dimensional Local Improvements to Trilinear Aggregation (LITA).
#
# The formula uses the odd Islam-Drevet/Pan-style trilinear-aggregation
# skeleton, removes disjoint local correction paths, and inserts rational
# 24->23, 42->40, and 60->57 local identities.  The sparse coefficient tables
# below are rational functions of d=(N+3)/2.
#
# The procedure supports odd square dimensions N > 18.
#------------------------------------------------------------------------------

LITAODDCOMPLEXITY:=proc(z)
    if type(z,numeric)
    then
        if not type(z,integer)
        then error "dimension should be an odd integer at least 19"
        end if:

        if z < 19 or irem(z,2)<>1
        then error "dimension should be an odd integer at least 19"
        end if:
    end if:

    return (4*z^3+57*z^2+14*z-15)/12-floor(3*(z-1)/8):
end proc:

LITAODD_BLOCK_DIMS:=proc(block,m)
    if block=1 then return [0,0,m,m]
    elif block=2 then return [0,m,m,m-1]
    elif block=3 then return [m,0,m-1,m]
    elif block=4 then return [m,m,m-1,m-1]
    else error "unknown block"
    end if:
end proc:

LITAODD_RAW:=proc(M,block,r,c,m)
local dims,ro,co,rows,cols:
    dims:=LITAODD_BLOCK_DIMS(block,m):
    ro:=dims[1]: co:=dims[2]: rows:=dims[3]: cols:=dims[4]:
    if r<0 or r>=rows or c<0 or c>=cols
    then return 0
    end if:
    return M[ro+r+1,co+c+1]:
end proc:

LITAODD_PROPSIZE:=proc(i,j,k,d)
local count:
    count:=0:
    if i=d-1 then count:=count+1 end if:
    if j=d-1 then count:=count+1 end if:
    if k=d-1 then count:=count+1 end if:
    return count<2:
end proc:

LITAODD_AENTRY:=proc(A,block,r,c,m)
local rr,cc,dims,rows,cols,row_sum,col_sum,total,t,s:
    rr:=r: cc:=c:
    if block=3 or block=4 then
        if rr=m-1 then return 0 end if:
        if rr=m then rr:=m-1 end if:
    end if:
    if block=2 or block=4 then
        if cc=m-1 then return 0 end if:
        if cc=m then cc:=m-1 end if:
    end if:

    dims:=LITAODD_BLOCK_DIMS(block,m):
    rows:=dims[3]: cols:=dims[4]:
    if rr<0 or rr>rows or cc<0 or cc>cols
    then return 0
    end if:

    row_sum:=add(LITAODD_RAW(A,block,rr,t,m),t=0..cols-1):
    col_sum:=add(LITAODD_RAW(A,block,t,cc,m),t=0..rows-1):
    total:=add(add(LITAODD_RAW(A,block,t,s,m),s=0..cols-1),t=0..rows-1):

    if rr<rows and cc<cols then return LITAODD_RAW(A,block,rr,cc,m)-row_sum/(cols+1)
    elif rr<rows and cc=cols then return -row_sum/(cols+1)
    elif rr=rows and cc<cols then return -col_sum+total/(cols+1)
    elif rr=rows and cc=cols then return total/(cols+1)
    else return 0
    end if:
end proc:

LITAODD_BENTRY:=proc(B,block,r,c,m)
local rr,cc,dims,rows,cols,row_sum,col_sum,total,t,s:
    rr:=r: cc:=c:
    if block=3 or block=4 then
        if rr=m-1 then return 0 end if:
        if rr=m then rr:=m-1 end if:
    end if:
    if block=2 or block=4 then
        if cc=m-1 then return 0 end if:
        if cc=m then cc:=m-1 end if:
    end if:

    dims:=LITAODD_BLOCK_DIMS(block,m):
    rows:=dims[3]: cols:=dims[4]:
    if rr<0 or rr>rows or cc<0 or cc>cols
    then return 0
    end if:

    row_sum:=add(LITAODD_RAW(B,block,rr,t,m),t=0..cols-1):
    col_sum:=add(LITAODD_RAW(B,block,t,cc,m),t=0..rows-1):
    total:=add(add(LITAODD_RAW(B,block,t,s,m),s=0..cols-1),t=0..rows-1):

    if rr<rows and cc<cols then return LITAODD_RAW(B,block,rr,cc,m)
    elif rr<rows and cc=cols then return -row_sum
    elif rr=rows and cc<cols then return -col_sum
    elif rr=rows and cc=cols then return total
    else return 0
    end if:
end proc:

LITAODD_CENTRY:=proc(C,block,r,c,m)
local dims,ro,co,rows,cols,row,col:
    dims:=LITAODD_BLOCK_DIMS(block,m):
    ro:=dims[1]: co:=dims[2]: rows:=dims[3]: cols:=dims[4]:
    if c<0 or c>=rows or r<0 or r>=cols
    then return 0
    end if:
    row:=ro+c: col:=co+r:
    # Transpose pullback from direct bilinear output coordinates.
    return C[col+1,row+1]:
end proc:

LITAODD_AF:=proc(A,codes,r,c,m)
local e,s,t:
    s:=0:
    for t from 1 to nops(codes) do
        e:=codes[t]:
        s:=s+e[1]*LITAODD_AENTRY(A,e[2],r,c,m):
    end do:
    return s:
end proc:

LITAODD_BF:=proc(B,codes,r,c,m)
local e,s,t:
    s:=0:
    for t from 1 to nops(codes) do
        e:=codes[t]:
        s:=s+e[1]*LITAODD_BENTRY(B,e[2],r,c,m):
    end do:
    return s:
end proc:

LITAODD_CF:=proc(C,codes,r,c,m)
local e,s,t:
    s:=0:
    for t from 1 to nops(codes) do
        e:=codes[t]:
        s:=s+e[1]*LITAODD_CENTRY(C,e[2],r,c,m):
    end do:
    return s:
end proc:

LITAODD_LABEL_FACTOR:=proc(A,B,C,label,axis,m,d)
local kind,call,i,j,k,ac,bc,cc,uc,vc,wc,xc,yc,zc,dc,inner:
    kind:=label[1]:

    if kind=s0 then
        call:=label[2]: i:=label[3]:
        if call=0 then
            ac:=[[1,2],[-1,1],[1,3]]:
            bc:=[[1,3],[1,2],[1,1]]:
            cc:=[[1,1],[-1,2],[1,3]]:
        else
            ac:=[[1,2],[1,3],[-1,4]]:
            bc:=[[1,4],[1,2],[1,3]]:
            cc:=[[1,2],[1,4],[-1,3]]:
        end if:
        if axis=1 then return LITAODD_AF(A,ac,i,i,m)
        elif axis=2 then return LITAODD_BF(B,bc,i,i,m)
        else return LITAODD_CF(C,cc,i,i,m)
        end if:
    end if:

    if kind=s1 then
        call:=label[2]: i:=label[3]: j:=label[4]: k:=label[5]:
        if call=0 then ac:=[[1,1]]: bc:=[[1,1]]: cc:=[[1,1]]:
        else ac:=[[1,4]]: bc:=[[1,4]]: cc:=[[1,4]]:
        end if:
        if axis=1 then
            return LITAODD_AF(A,ac,i,j,m)+LITAODD_AF(A,ac,j,k,m)+LITAODD_AF(A,ac,k,i,m)
        elif axis=2 then
            return LITAODD_BF(B,bc,j,k,m)+LITAODD_BF(B,bc,k,i,m)+LITAODD_BF(B,bc,i,j,m)
        else
            return LITAODD_CF(C,cc,k,i,m)+LITAODD_CF(C,cc,i,j,m)+LITAODD_CF(C,cc,j,k,m)
        end if:
    end if:

    if kind=s2 then
        call:=label[2]: i:=label[3]: j:=label[4]: k:=label[5]:
        if call=0 then
            ac:=[[-1,1]]: bc:=[[1,2]]: cc:=[[-1,2]]:
            uc:=[[1,3]]: vc:=[[1,1]]: wc:=[[1,3]]:
            xc:=[[1,2]]: yc:=[[1,3]]: zc:=[[1,1]]:
        else
            ac:=[[-1,4]]: bc:=[[1,3]]: cc:=[[-1,3]]:
            uc:=[[1,2]]: vc:=[[1,4]]: wc:=[[1,2]]:
            xc:=[[1,3]]: yc:=[[1,2]]: zc:=[[1,4]]:
        end if:
        if axis=1 then
            return LITAODD_AF(A,ac,i,j,m)+LITAODD_AF(A,uc,j,k,m)+LITAODD_AF(A,xc,k,i,m)
        elif axis=2 then
            return LITAODD_BF(B,bc,j,k,m)+LITAODD_BF(B,vc,k,i,m)+LITAODD_BF(B,yc,i,j,m)
        else
            return LITAODD_CF(C,cc,k,i,m)+LITAODD_CF(C,wc,i,j,m)+LITAODD_CF(C,zc,j,k,m)
        end if:
    end if:

    if kind=u2p then
        call:=label[2]: i:=label[3]:
        if call=0 then
            ac:=[[1,1]]:
        else
            ac:=[[1,4]]:
        end if:
        inner:=(d-9)*LITAODD_CF(C,ac,i,i,m)
            +add(LITAODD_CF(C,ac,k,i,m)+LITAODD_CF(C,ac,i,k,m),k=0..d-1):
        if axis=1 then return LITAODD_AF(A,ac,i,i,m)
        elif axis=2 then return LITAODD_BF(B,ac,i,i,m)
        else return -inner
        end if:
    end if:

    if kind=u2 then
        call:=label[2]: i:=label[3]:
        if call=0 then ac:=[[-1,1]]: bc:=[[1,3]]: dc:=[[-1,2]]: wc:=[[1,3]]: zc:=[[1,1]]:
        elif call=1 then ac:=[[1,2]]: bc:=[[1,1]]: dc:=[[1,1]]: wc:=[[-1,2]]: zc:=[[1,3]]:
        elif call=2 then ac:=[[1,3]]: bc:=[[1,2]]: dc:=[[1,3]]: wc:=[[1,1]]: zc:=[[-1,2]]:
        elif call=3 then ac:=[[-1,4]]: bc:=[[1,2]]: dc:=[[-1,3]]: wc:=[[1,2]]: zc:=[[1,4]]:
        elif call=4 then ac:=[[1,3]]: bc:=[[1,4]]: dc:=[[1,4]]: wc:=[[-1,3]]: zc:=[[1,2]]:
        else ac:=[[1,2]]: bc:=[[1,3]]: dc:=[[1,2]]: wc:=[[1,4]]: zc:=[[-1,3]]:
        end if:
        inner:=add(LITAODD_CF(C,dc,k,i,m),k=0..d-1)
            +d*LITAODD_CF(C,wc,i,i,m)
            +add(LITAODD_CF(C,zc,i,k,m),k=0..d-1):
        if axis=1 then return LITAODD_AF(A,ac,i,i,m)
        elif axis=2 then return LITAODD_BF(B,bc,i,i,m)
        else return -inner
        end if:
    end if:

    if kind=u1 then
        call:=label[2]: i:=label[3]: j:=label[4]:
        if call=0 then ac:=[[1,1]]: bc:=[[1,1]]: cc:=[[1,1]]: wc:=[[1,1]]: zc:=[[1,1]]:
        elif call=1 then ac:=[[-1,1]]: bc:=[[1,3]]: cc:=[[-1,2]]: wc:=[[1,3]]: zc:=[[1,1]]:
        elif call=2 then ac:=[[1,2]]: bc:=[[1,3]]: cc:=[[1,2]]: wc:=[[1,4]]: zc:=[[-1,3]]:
        elif call=3 then ac:=[[-1,4]]: bc:=[[1,2]]: cc:=[[-1,3]]: wc:=[[1,2]]: zc:=[[1,4]]:
        elif call=4 then ac:=[[1,2]]: bc:=[[1,1]]: cc:=[[1,1]]: wc:=[[-1,2]]: zc:=[[1,3]]:
        elif call=5 then ac:=[[1,3]]: bc:=[[1,2]]: cc:=[[1,3]]: wc:=[[1,1]]: zc:=[[-1,2]]:
        elif call=6 then ac:=[[1,3]]: bc:=[[1,4]]: cc:=[[1,4]]: wc:=[[-1,3]]: zc:=[[1,2]]:
        else ac:=[[1,4]]: bc:=[[1,4]]: cc:=[[1,4]]: wc:=[[1,4]]: zc:=[[1,4]]:
        end if:
        inner:=d*LITAODD_CF(C,wc,i,j,m)
            +add(LITAODD_CF(C,cc,k,i,m)+LITAODD_CF(C,zc,j,k,m),k=0..d-1):
        if axis=1 then return LITAODD_AF(A,ac,i,j,m)
        elif axis=2 then return LITAODD_BF(B,bc,i,j,m)
        else return -inner
        end if:
    end if:

    if kind=u3 then
        call:=label[2]: i:=label[3]:
        if call=0 then ac:=[[1,1],[1,2]]: yc:=[[1,1]]: cc:=[[1,1]]:
        elif call=1 then ac:=[[1,1],[1,2]]: yc:=[[1,3]]: cc:=[[1,2]]:
        elif call=2 then ac:=[[1,3],[1,4]]: yc:=[[1,2]]: cc:=[[1,3]]:
        else ac:=[[1,3],[1,4]]: yc:=[[1,4]]: cc:=[[1,4]]:
        end if:
        inner:=add(LITAODD_CF(C,cc,k,i,m),k=0..d-1):
        if axis=1 then return LITAODD_AF(A,ac,i,d-1,m)
        elif axis=2 then return LITAODD_BF(B,yc,i,d-1,m)
        else return -inner
        end if:
    end if:

    if kind=u4 then
        call:=label[2]: j:=label[3]:
        if call=0 then ac:=[[1,1]]: yc:=[[1,1],[-1,3]]: zc:=[[1,1]]:
        elif call=1 then ac:=[[1,2]]: yc:=[[1,1],[-1,3]]: zc:=[[1,3]]:
        elif call=2 then ac:=[[1,3]]: yc:=[[-1,2],[1,4]]: zc:=[[1,2]]:
        else ac:=[[1,4]]: yc:=[[-1,2],[1,4]]: zc:=[[1,4]]:
        end if:
        inner:=add(LITAODD_CF(C,zc,j,k,m),k=0..d-1):
        if axis=1 then return LITAODD_AF(A,ac,d-1,j,m)
        elif axis=2 then return LITAODD_BF(B,yc,d-1,j,m)
        else return -inner
        end if:
    end if:

    error "unknown label kind":
end proc:

LITAODD_LABEL_TERM:=proc(A,B,C,label,m,d)
    return LITAODD_LABEL_FACTOR(A,B,C,label,1,m,d)
        *LITAODD_LABEL_FACTOR(A,B,C,label,2,m,d)
        *LITAODD_LABEL_FACTOR(A,B,C,label,3,m,d):
end proc:

LITAODD_GROUP_FROM_TARGET:=proc(label,z)
    if label[1]=s1 and label[2]=1 then
        if label[3]=label[4] and label[5]=z then
            return [label,[u2p,1,label[3]]]
        elif label[5]=z then
            return [label,[u1,7,label[3],label[4]]]
        elif label[3]=z then
            return [label,[u1,7,label[4],label[5]]]
        end if:
    end if:
    return [label]:
end proc:

LITAODD_GROUP_FACTOR:=proc(A,B,C,group,axis,m,d)
local t,s:
    if axis=1 or axis=2 then
        return LITAODD_LABEL_FACTOR(A,B,C,group[1],axis,m,d)
    end if:
    s:=0:
    for t from 1 to nops(group) do
        s:=s+LITAODD_LABEL_FACTOR(A,B,C,group[t],3,m,d):
    end do:
    return s:
end proc:

LITAODD_GROUP_TERM:=proc(A,B,C,group,m,d)
local t,s:
    s:=0:
    for t from 1 to nops(group) do
        s:=s+LITAODD_LABEL_TERM(A,B,C,group[t],m,d):
    end do:
    return s:
end proc:

LITAODD_CYCLE_TARGETS:=proc(first,second,z)
    return [[s1,1,first,first,z], [u2,4,first], [u2,2,first],
        [s2,1,first,first,z], [u2,3,first], [s2,1,first,z,first],
        [s1,1,first,second,z], [s2,1,second,z,first],
        [u1,6,first,second], [u1,5,first,second],
        [s2,1,first,second,z], [u1,3,first,second],
        [s1,1,second,second,z], [s2,1,second,z,second],
        [u2,4,second], [u2,2,second], [u2,3,second],
        [s2,1,second,second,z], [s1,1,z,second,first],
        [u1,3,second,first], [u1,5,second,first],
        [u1,6,second,first], [s2,1,second,first,z],
        [s2,1,first,z,second]
    ]:
end proc:

LITAODD_ORDERED_CYCLE_GROUPS:=proc(first,second,z)
local targets,t,groups:
    targets:=LITAODD_CYCLE_TARGETS(first,second,z):
    groups:=[]:
    for t from 1 to nops(targets) do
        groups:=[op(groups),LITAODD_GROUP_FROM_TARGET(targets[t],z)]:
    end do:
    return groups:
end proc:

LITAODD_APPEND_GROUP:=proc(out,group)
    if member(group,out) then return out end if:
    return [op(out),group]:
end proc:

LITAODD_APPEND_RANGE:=proc(out,groups,a,b)
local t,res:
    res:=out:
    for t from a to b do res:=LITAODD_APPEND_GROUP(res,groups[t]) end do:
    return res:
end proc:

LITAODD_ORDERED_PATH3_GROUPS:=proc(indices,z)
local left,right,out:
    left:=LITAODD_ORDERED_CYCLE_GROUPS(indices[1],indices[2],z):
    right:=LITAODD_ORDERED_CYCLE_GROUPS(indices[2],indices[3],z):
    out:=[]:
    out:=LITAODD_APPEND_RANGE(out,left,1,6):
    out:=LITAODD_APPEND_RANGE(out,left,7,12):
    out:=LITAODD_APPEND_RANGE(out,left,13,18):
    out:=LITAODD_APPEND_RANGE(out,left,19,24):
    out:=LITAODD_APPEND_RANGE(out,right,7,12):
    out:=LITAODD_APPEND_RANGE(out,right,13,18):
    out:=LITAODD_APPEND_RANGE(out,right,19,24):
    if nops(out)<>42 then error "expected 42 path3 groups" end if:
    return out:
end proc:

LITAODD_ORDERED_PATH4_GROUPS:=proc(indices,z)
local first_cycle,cycle,out,t:
    first_cycle:=LITAODD_ORDERED_CYCLE_GROUPS(indices[1],indices[2],z):
    out:=LITAODD_APPEND_RANGE([],first_cycle,1,6):
    for t from 1 to nops(indices)-1 do
        cycle:=LITAODD_ORDERED_CYCLE_GROUPS(indices[t],indices[t+1],z):
        out:=LITAODD_APPEND_RANGE(out,cycle,7,12):
        out:=LITAODD_APPEND_RANGE(out,cycle,13,18):
        out:=LITAODD_APPEND_RANGE(out,cycle,19,24):
    end do:
    if nops(out)<>60 then error "expected 60 path4 groups" end if:
    return out:
end proc:

LITAODD_INDEX_BLOCKS:=proc(m)
local regular,q,rem,lengths,t,blocks,start,ell,p:
    regular:=m-1:
    q:=iquo(regular,4): rem:=irem(regular,4):
    lengths:=[]:
    if rem=0 then
        for t from 1 to q do lengths:=[op(lengths),4] end do:
    elif rem=1 then
        for t from 1 to q-1 do lengths:=[op(lengths),4] end do:
        lengths:=[op(lengths),3,2]:
    elif rem=2 then
        for t from 1 to q do lengths:=[op(lengths),4] end do:
        lengths:=[op(lengths),2]:
    else
        for t from 1 to q do lengths:=[op(lengths),4] end do:
        lengths:=[op(lengths),3]:
    end if:

    blocks:=[]: start:=0:
    for p from 1 to nops(lengths) do
        ell:=lengths[p]:
        blocks:=[op(blocks),[seq(start+t,t=0..ell-1)]]:
        start:=start+ell:
    end do:
    if start<>regular then error "block partition does not cover all regular indices" end if:
    return blocks:
end proc:

LITAODD_LOCAL_FORM:=proc(A,B,C,groups,entries,term,axis,m,d)
local e,s,t:
    s:=0:
    for t from 1 to nops(entries) do
        e:=entries[t]:
        if e[2]=term then
            s:=s+e[3]*LITAODD_GROUP_FACTOR(A,B,C,groups[e[1]],axis,m,d):
        end if:
    end do:
    return s:
end proc:

LITAODD_LOCAL_SUM:=proc(A,B,C,groups,utab,vtab,wtab,rk,m,d)
local r,fu,fv,fw,s:
    s:=0:
    for r from 1 to rk do
        fu:=LITAODD_LOCAL_FORM(A,B,C,groups,utab,r,1,m,d):
        fv:=LITAODD_LOCAL_FORM(A,B,C,groups,vtab,r,2,m,d):
        fw:=LITAODD_LOCAL_FORM(A,B,C,groups,wtab,r,3,m,d):
        s:=s+fu*fv*fw:
    end do:
    return s:
end proc:

LITAODD_REMOVE_SUM:=proc(A,B,C,groups,m,d)
local g,s,t:
    s:=0:
    for t from 1 to nops(groups) do
        g:=groups[t]:
        s:=s+LITAODD_GROUP_TERM(A,B,C,g,m,d):
    end do:
    return s:
end proc:

LITAODD_PATH4U:=proc(d)
    return [[1,5,-1], [1,9,1], [1,10,-1], [1,51,-1],
        [2,1,-1], [2,2,-1], [2,3,-1], [2,4,-d],
        [2,5,-d], [2,6,-1], [2,8,1], [2,49,-1],
        [2,51,1], [6,1,1], [6,2,1], [6,3,1],
        [6,4,1], [6,5,1], [6,6,1], [6,49,1],
        [7,3,-1], [7,33,-1], [7,34,-1], [7,49,1/2],
        [8,1,-1], [8,2,-(d - 1)/d], [8,3,-(d - 1)/d], [8,49,-1],
        [13,18,1], [13,34,-1], [13,42,1], [14,7,1],
        [14,11,1/d], [14,12,1/d], [14,15,1], [14,16,1],
        [14,17,1], [14,18,1], [14,19,1], [14,52,1],
        [15,7,-1], [15,11,-1/d], [15,12,-1/d], [15,15,-1],
        [15,16,-1], [15,17,-d], [15,18,-d], [15,19,-1],
        [15,43,1], [15,52,-1], [19,9,1], [19,12,1],
        [19,14,1], [19,50,-2], [21,11,-(d - 1)/d], [21,12,-(d - 1)/d],
        [21,13,-1], [21,50,-1], [25,16,1], [25,20,1],
        [25,21,1], [25,52,-2], [26,7,-1], [26,15,-(d - 1)/d],
        [26,16,-(d - 1)/d], [26,52,-1], [31,20,1], [31,23,1],
        [31,25,1], [31,54,-2], [32,22,-1/(d - 1)], [32,23,-1/(d - 1)],
        [32,35,-1/(d - 1)], [32,36,-1/(d - 1)], [32,39,d], [32,40,d],
        [32,41,1], [32,46,-1], [32,57,-1], [33,22,d/(d - 1)],
        [33,23,d/(d - 1)], [33,24,1], [33,35,1/(d - 1)], [33,36,1/(d - 1)],
        [33,39,-d], [33,40,-d], [33,41,-1], [33,46,1],
        [33,54,1], [33,57,1], [37,34,-1], [37,36,1],
        [37,38,1], [37,53,-2], [39,35,1], [39,36,1],
        [39,37,1], [39,53,1], [43,40,1], [43,44,1],
        [43,45,1], [43,57,1], [44,39,1 - d], [44,40,1 - d],
        [44,46,1], [44,57,1], [49,31,1], [49,44,1],
        [49,47,1], [49,56,1], [50,26,1/d], [50,27,1/d],
        [50,30,1], [50,31,1], [50,32,1], [51,26,-1/d],
        [51,27,-1/d], [51,30,-d], [51,31,-d], [51,32,-1],
        [51,48,1], [51,56,-1/2], [55,20,1], [55,27,1],
        [55,29,1], [55,55,4], [57,26,-(d - 1)/d], [57,27,-(d - 1)/d],
        [57,28,-1], [57,55,-1]]:
end proc:

LITAODD_PATH4V:=proc(d)
    return [[1,4,1], [1,5,1], [1,6,1], [1,51,1],
        [3,8,1], [3,9,-1], [3,10,-d], [3,14,-1/d],
        [3,51,1], [4,9,1], [4,10,1], [4,14,1/d],
        [7,2,1], [7,3,1], [7,6,1], [7,49,1],
        [10,1,1], [10,33,-d], [10,34,-1], [10,38,-1/d],
        [10,42,-1/d], [10,49,2], [11,33,1], [11,34,1],
        [11,38,1/d], [11,42,1/d], [13,17,1], [13,18,1],
        [13,19,1], [16,42,-(d - 1)/d], [16,43,-1], [19,11,1],
        [19,12,1], [19,19,1], [19,50,2], [20,13,-1],
        [20,14,-(d - 1)/d], [20,50,-1], [25,15,1], [25,16,1],
        [25,19,1], [25,52,1], [28,7,1], [28,20,-1],
        [28,21,-d], [28,25,-1/d], [28,29,-1/d], [28,52,1],
        [29,20,1], [29,21,1], [29,25,1/d], [29,29,1/d],
        [31,22,1], [31,23,1], [31,41,1], [31,54,1],
        [34,24,-1], [34,25,-(d - 1)/d], [34,54,-1], [37,35,1],
        [37,36,1], [37,41,1], [37,53,1], [38,37,-1],
        [38,38,-(d - 1)/d], [38,53,-1], [43,39,1], [43,40,1],
        [43,41,1], [43,57,1], [46,44,-1], [46,45,-d],
        [46,46,1], [46,47,-1/d], [46,57,2], [47,44,1],
        [47,45,1], [47,47,1/d], [49,30,1], [49,31,1],
        [49,32,1], [49,56,1], [52,47,-(d - 1)/d], [52,48,-1],
        [52,56,-1], [55,26,1], [55,27,1], [55,32,1],
        [55,55,2], [56,28,-1], [56,29,-(d - 1)/d], [56,55,-1]]:
end proc:

LITAODD_PATH4W:=proc(d)
    return [[1,4,1], [1,5,-1], [2,4,1], [2,5,-1],
        [2,8,-1], [2,9,1/(d - 1)], [2,10,1/(d - 1)], [2,11,3],
        [2,12,-2], [2,13,1/2], [2,14,-d/(d - 1)], [2,50,-1/2],
        [2,51,1], [3,8,1], [4,9,-d/(d - 1)], [4,10,-1/(d - 1)],
        [4,14,d/(d - 1)], [5,9,-1/(d - 1)], [5,10,-1/(d - 1)], [5,14,1/(d - 1)],
        [6,4,d + 1], [6,5,-d], [6,8,-d], [6,9,d/(d - 1)],
        [6,10,d/(d - 1)], [6,11,3*d], [6,12,-2*d], [6,13,d/2],
        [6,14,-d^2/(d - 1)], [6,50,-d/2], [6,51,d], [7,2,1],
        [7,3,-1], [8,2,-d/(d - 1)], [8,4,-d^2/(d - 1)], [8,5,d],
        [8,6,d/(d - 1)], [8,8,d], [8,9,-d/(d - 1)], [8,10,-d/(d - 1)],
        [8,11,-3*d], [8,12,2*d], [8,13,-d/2], [8,14,d^2/(d - 1)],
        [8,50,d/2], [8,51,-d], [9,2,-d/(d - 1)], [9,4,-d/(d - 1)],
        [9,5,1], [9,6,1/(d - 1)], [9,8,1], [9,9,-1/(d - 1)],
        [9,10,-1/(d - 1)], [9,11,-3], [9,12,2], [9,13,-1/2],
        [9,14,d/(d - 1)], [9,50,1/2], [9,51,-1], [10,1,-1],
        [11,1,2*d], [11,2,d*(3*d - 1)/(2*(d - 1))], [11,3,-d/2], [11,4,d^2/(d - 1)],
        [11,5,-d], [11,6,-d/(d - 1)], [11,8,-d], [11,9,d/(d - 1)],
        [11,10,d/(d - 1)], [11,14,-d/(d - 1)], [11,33,1], [11,49,-d],
        [11,51,d], [12,1,2], [12,2,(3*d - 1)/(2*(d - 1))], [12,3,-1/2],
        [12,4,d/(d - 1)], [12,5,-1], [12,6,-1/(d - 1)], [12,8,-1],
        [12,9,1/(d - 1)], [12,10,1/(d - 1)], [12,14,-1/(d - 1)], [12,49,-1],
        [12,51,1], [13,17,-1], [13,18,1], [14,7,-d],
        [14,11,-d/(d - 1)], [14,15,-d*(3*d - 2)/(d - 1)], [14,16,2*d], [14,17,-1/(d - 1)],
        [14,19,d/(d - 1)], [14,20,2*d/(d - 1)], [14,21,-2*d/(d - 1)], [14,22,3*d],
        [14,23,-2*d], [14,24,d], [14,25,-2*d^2/(d - 1)], [14,26,d*(3*d - 4)/(d - 1)],
        [14,27,-4*d], [14,28,-d/2], [14,29,-2*d^2/(d - 1)], [14,30,d*(2*d - 3)/(d - 1)],
        [14,31,-2*d], [14,32,d/(d - 1)], [14,35,1], [14,39,-d^2/(d - 1)],
        [14,40,d], [14,41,d/(d - 1)], [14,44,2*d/(d - 1)], [14,45,-2*d/(d - 1)],
        [14,46,2*d], [14,47,-2*d^2/(d - 1)], [14,48,d], [14,52,d],
        [14,54,-d], [14,55,d/2], [14,56,2*d], [14,57,-d],
        [15,7,-1], [15,11,-1/(d - 1)], [15,15,-(3*d - 2)/(d - 1)], [15,16,2],
        [15,17,-1/(d - 1)], [15,19,1/(d - 1)], [15,20,2/(d - 1)], [15,21,-2/(d - 1)],
        [15,22,3], [15,23,-2], [15,24,1], [15,25,-2*d/(d - 1)],
        [15,26,(3*d - 4)/(d - 1)], [15,27,-4], [15,28,-1/2], [15,29,-2*d/(d - 1)],
        [15,30,(2*d - 3)/(d - 1)], [15,31,-2], [15,32,1/(d - 1)], [15,35,1/d],
        [15,39,-d/(d - 1)], [15,40,1], [15,41,1/(d - 1)], [15,44,2/(d - 1)],
        [15,45,-2/(d - 1)], [15,46,2], [15,47,-2*d/(d - 1)], [15,48,1],
        [15,52,1], [15,54,-1], [15,55,1/2], [15,56,2],
        [15,57,-1], [16,43,-1], [17,1,-2], [17,2,-(3*d - 1)/(2*(d - 1))],
        [17,3,1/2], [17,4,-d/(d - 1)], [17,5,1], [17,6,1/(d - 1)],
        [17,8,1], [17,9,-1/(d - 1)], [17,10,-1/(d - 1)], [17,14,1/(d - 1)],
        [17,33,-1/(d - 1)], [17,34,1/(d - 1)], [17,38,1/(d - 1)], [17,42,d/(d - 1)],
        [17,49,1], [17,51,-1], [18,1,-2*d], [18,2,-d*(3*d - 1)/(2*(d - 1))],
        [18,3,d/2], [18,4,-d^2/(d - 1)], [18,5,d], [18,6,d/(d - 1)],
        [18,8,d], [18,9,-d/(d - 1)], [18,10,-d/(d - 1)], [18,14,d/(d - 1)],
        [18,33,-d/(d - 1)], [18,34,d/(d - 1)], [18,38,d/(d - 1)], [18,42,d/(d - 1)],
        [18,49,d], [18,51,-d], [19,11,-1], [19,12,1],
        [20,14,1], [21,13,1], [22,11,-1], [25,15,-1],
        [25,16,1], [26,7,d], [26,15,3*d], [26,16,-2*d],
        [26,20,-2*d/(d - 1)], [26,21,2*d/(d - 1)], [26,22,-3*d], [26,23,2*d],
        [26,24,-d], [26,25,2*d^2/(d - 1)], [26,26,-d*(3*d - 4)/(d - 1)], [26,27,4*d],
        [26,28,d/2], [26,29,2*d^2/(d - 1)], [26,30,-d*(2*d - 3)/(d - 1)], [26,31,2*d],
        [26,32,-d/(d - 1)], [26,35,-1], [26,39,d^2/(d - 1)], [26,40,-d],
        [26,41,-d/(d - 1)], [26,44,-2*d/(d - 1)], [26,45,2*d/(d - 1)], [26,46,-2*d],
        [26,47,2*d^2/(d - 1)], [26,48,-d], [26,52,-d], [26,54,d],
        [26,55,-d/2], [26,56,-2*d], [26,57,d], [27,7,1],
        [27,15,2], [27,16,-2], [27,20,-2/(d - 1)], [27,21,2/(d - 1)],
        [27,22,-3], [27,23,2], [27,24,-1], [27,25,2*d/(d - 1)],
        [27,26,-(3*d - 4)/(d - 1)], [27,27,4], [27,28,1/2], [27,29,2*d/(d - 1)],
        [27,30,-(2*d - 3)/(d - 1)], [27,31,2], [27,32,-1/(d - 1)], [27,35,-1/d],
        [27,39,d/(d - 1)], [27,40,-1], [27,41,-1/(d - 1)], [27,44,-2/(d - 1)],
        [27,45,2/(d - 1)], [27,46,-2], [27,47,2*d/(d - 1)], [27,48,-1],
        [27,52,-1], [27,54,1], [27,55,-1/2], [27,56,-2],
        [27,57,1], [28,7,-1], [29,20,-d/(d - 1)], [29,21,1/(d - 1)],
        [29,22,-3*d/2], [29,23,d], [29,24,-d/2], [29,25,d^2/(d - 1)],
        [29,26,-d*(3*d - 4)/(2*(d - 1))], [29,27,2*d], [29,28,d/4], [29,29,d^2/(d - 1)],
        [29,30,-d*(2*d - 3)/(2*(d - 1))], [29,31,d], [29,32,-d/(2*(d - 1))], [29,35,-3*d/2],
        [29,36,d], [29,37,-d/2], [29,38,d], [29,39,d^2/(2*(d - 1))],
        [29,40,-d/2], [29,41,-d/(2*(d - 1))], [29,44,-d/(d - 1)], [29,45,d/(d - 1)],
        [29,46,-d], [29,47,d^2/(d - 1)], [29,48,-d/2], [29,53,d/2],
        [29,54,d/2], [29,55,-d/4], [29,56,-d], [29,57,d/2],
        [30,20,-1/(d - 1)], [30,21,1/(d - 1)], [30,22,-3/2], [30,23,1],
        [30,24,-1/2], [30,25,d/(d - 1)], [30,26,-(3*d - 4)/(2*(d - 1))], [30,27,2],
        [30,28,1/4], [30,29,d/(d - 1)], [30,30,-(2*d - 3)/(2*(d - 1))], [30,31,1],
        [30,32,-1/(2*(d - 1))], [30,35,-3/2], [30,36,1], [30,37,-1/2],
        [30,38,1], [30,39,d/(2*(d - 1))], [30,40,-1/2], [30,41,-1/(2*(d - 1))],
        [30,44,-1/(d - 1)], [30,45,1/(d - 1)], [30,46,-1], [30,47,d/(d - 1)],
        [30,48,-1/2], [30,53,1/2], [30,54,1/2], [30,55,-1/4],
        [30,56,-1], [30,57,1/2], [31,22,-1], [31,23,1],
        [32,22,1], [32,26,-d/(d - 1)], [32,30,d*(2*d - 3)/(d - 1)], [32,31,-2*d],
        [32,32,d/(d - 1)], [32,35,1], [32,39,-d^2/(d - 1)], [32,40,d],
        [32,41,d/(d - 1)], [32,44,2*d/(d - 1)], [32,45,-2*d/(d - 1)], [32,46,2*d],
        [32,47,-2*d^2/(d - 1)], [32,48,d], [32,56,2*d], [32,57,-d],
        [33,22,1], [33,26,-1/(d - 1)], [33,30,(2*d - 3)/(d - 1)], [33,31,-2],
        [33,32,1/(d - 1)], [33,35,1/d], [33,39,-d/(d - 1)], [33,40,1],
        [33,41,1/(d - 1)], [33,44,2/(d - 1)], [33,45,-2/(d - 1)], [33,46,2],
        [33,47,-2*d/(d - 1)], [33,48,1], [33,56,2], [33,57,-1],
        [34,24,-1], [35,22,3/2], [35,23,-1], [35,24,1/2],
        [35,26,(3*d - 4)/(2*(d - 1))], [35,27,-2], [35,28,-1/4], [35,29,-1],
        [35,30,(2*d - 3)/(2*(d - 1))], [35,31,-1], [35,32,1/(2*(d - 1))], [35,35,3/2],
        [35,36,-1], [35,37,1/2], [35,38,-1], [35,39,-d/(2*(d - 1))],
        [35,40,1/2], [35,41,1/(2*(d - 1))], [35,44,1/(d - 1)], [35,45,-1/(d - 1)],
        [35,46,1], [35,47,-d/(d - 1)], [35,48,1/2], [35,53,-1/2],
        [35,54,-1/2], [35,55,1/4], [35,56,1], [35,57,-1/2],
        [36,22,3*d/2], [36,23,-d], [36,24,d/2], [36,25,-d],
        [36,26,d*(3*d - 4)/(2*(d - 1))], [36,27,-2*d], [36,28,-d/4], [36,29,-d],
        [36,30,d*(2*d - 3)/(2*(d - 1))], [36,31,-d], [36,32,d/(2*(d - 1))], [36,35,3*d/2],
        [36,36,-d], [36,37,d/2], [36,38,-d], [36,39,-d^2/(2*(d - 1))],
        [36,40,d/2], [36,41,d/(2*(d - 1))], [36,44,d/(d - 1)], [36,45,-d/(d - 1)],
        [36,46,d], [36,47,-d^2/(d - 1)], [36,48,d/2], [36,53,-d/2],
        [36,54,-d/2], [36,55,d/4], [36,56,d], [36,57,-d/2],
        [37,35,-1], [37,36,1], [38,38,1], [39,37,-1],
        [40,35,(d - 1)/d], [43,39,-1], [43,40,1], [44,26,d/(d - 1)],
        [44,30,-d*(2*d - 3)/(d - 1)], [44,31,2*d], [44,32,-d/(d - 1)], [44,39,d + 1],
        [44,40,-d], [44,44,-2*d/(d - 1)], [44,45,2*d/(d - 1)], [44,46,-2*d],
        [44,47,2*d^2/(d - 1)], [44,48,-d], [44,56,-2*d], [44,57,d],
        [45,26,1/(d - 1)], [45,30,-(2*d - 3)/(d - 1)], [45,31,2], [45,32,-1/(d - 1)],
        [45,39,1], [45,40,-1], [45,44,-2/(d - 1)], [45,45,2/(d - 1)],
        [45,46,-2], [45,47,2*d/(d - 1)], [45,48,-1], [45,56,-2],
        [45,57,1], [46,46,1], [47,26,-d*(3*d - 4)/(2*(d - 1))], [47,27,2*d],
        [47,28,d/4], [47,29,d], [47,30,-d*(2*d - 3)/(2*(d - 1))], [47,31,d],
        [47,32,-d/(2*(d - 1))], [47,44,-d/(d - 1)], [47,45,1/(d - 1)], [47,47,d^2/(d - 1)],
        [47,48,-d/2], [47,55,-d/4], [47,56,-d], [48,26,-(3*d - 4)/(2*(d - 1))],
        [48,27,2], [48,28,1/4], [48,29,1], [48,30,-(2*d - 3)/(2*(d - 1))],
        [48,31,1], [48,32,-1/(2*(d - 1))], [48,44,-1/(d - 1)], [48,45,1/(d - 1)],
        [48,47,d/(d - 1)], [48,48,-1/2], [48,55,-1/4], [48,56,-1],
        [49,30,-1], [49,31,1], [50,26,-d/(d - 1)], [50,30,-1/(d - 1)],
        [50,32,d/(d - 1)], [51,26,-1/(d - 1)], [51,30,-1/(d - 1)], [51,32,1/(d - 1)],
        [52,48,-1], [53,26,(3*d - 4)/(2*(d - 1))], [53,27,-2], [53,28,-1/4],
        [53,29,-1], [53,30,(2*d - 3)/(2*(d - 1))], [53,31,-1], [53,32,1/(2*(d - 1))],
        [53,48,1/2], [53,55,1/4], [53,56,1], [54,26,d*(3*d - 4)/(2*(d - 1))],
        [54,27,-2*d], [54,28,-d/4], [54,29,-d], [54,30,d*(2*d - 3)/(2*(d - 1))],
        [54,31,-d], [54,32,d/(2*(d - 1))], [54,47,-d], [54,48,d/2],
        [54,55,d/4], [54,56,d], [55,26,-1], [55,27,1],
        [56,29,1], [57,28,1], [58,26,-1]]:
end proc:

LITAODD_PATH3U:=proc(d)
    return [[1,5,-1], [1,9,1], [1,10,-1], [1,37,-1],
        [2,1,-1], [2,2,-1], [2,3,-1], [2,4,-d],
        [2,5,-d], [2,6,-1], [2,8,1], [2,35,-1],
        [2,37,1], [6,1,1], [6,2,1], [6,3,1],
        [6,4,1], [6,5,1], [6,6,1], [6,35,1],
        [7,3,-1], [7,26,-1], [7,27,-1], [7,35,1/2],
        [8,1,-1], [8,2,-(d - 1)/d], [8,3,-(d - 1)/d], [8,35,-1],
        [13,18,1], [13,27,-1], [13,33,1], [14,7,1],
        [14,11,1/d], [14,12,1/d], [14,15,1], [14,16,1],
        [14,17,1], [14,18,1], [14,19,1], [14,38,1],
        [15,7,-1], [15,11,-1/d], [15,12,-1/d], [15,15,-1],
        [15,16,-1], [15,17,-d], [15,18,-d], [15,19,-1],
        [15,34,1], [15,38,-1], [19,9,1], [19,12,1],
        [19,14,1], [19,36,-2], [21,11,-(d - 1)/d], [21,12,-(d - 1)/d],
        [21,13,-1], [21,36,-1], [25,16,1], [25,20,1],
        [25,21,1], [25,38,-2], [26,7,-1], [26,15,-(d - 1)/d],
        [26,16,-(d - 1)/d], [26,38,-1], [31,20,1], [31,23,1],
        [31,25,1], [31,40,-2], [32,22,-1/(d - 1)], [32,23,-1/(d - 1)],
        [32,28,-1/(d - 1)], [32,29,-1/(d - 1)], [32,32,1], [33,22,d/(d - 1)],
        [33,23,d/(d - 1)], [33,24,1], [33,28,1/(d - 1)], [33,29,1/(d - 1)],
        [33,32,-1], [33,40,1], [37,27,-1], [37,29,1],
        [37,31,1], [37,39,-2], [39,28,1], [39,29,1],
        [39,30,1], [39,39,1]]:
end proc:

LITAODD_PATH3V:=proc(d)
    return [[1,4,1], [1,5,1], [1,6,1], [1,37,1],
        [3,8,1], [3,9,-1], [3,10,-d], [3,14,-1/d],
        [3,37,1], [4,9,1], [4,10,1], [4,14,1/d],
        [7,2,1], [7,3,1], [7,6,1], [7,35,1],
        [10,1,1], [10,26,-d], [10,27,-1], [10,31,-1/d],
        [10,33,-1/d], [10,35,2], [11,26,1], [11,27,1],
        [11,31,1/d], [11,33,1/d], [13,17,1], [13,18,1],
        [13,19,1], [16,33,-(d - 1)/d], [16,34,-1], [19,11,1],
        [19,12,1], [19,19,1], [19,36,2], [20,13,-1],
        [20,14,-(d - 1)/d], [20,36,-1], [25,15,1], [25,16,1],
        [25,19,1], [25,38,1], [28,7,1], [28,20,-1],
        [28,21,-d], [28,25,-1/d], [28,38,1], [29,20,1],
        [29,21,1], [29,25,1/d], [31,22,1], [31,23,1],
        [31,32,1], [31,40,1], [34,24,-1], [34,25,-(d - 1)/d],
        [34,40,-1], [37,28,1], [37,29,1], [37,32,1],
        [37,39,1], [38,30,-1], [38,31,-(d - 1)/d], [38,39,-1]]:
end proc:

LITAODD_PATH3W:=proc(d)
    return [[1,4,1], [1,5,-1], [2,4,1], [2,5,-1],
        [2,8,-1], [2,9,1/(d - 1)], [2,10,1/(d - 1)], [2,11,3],
        [2,12,-2], [2,13,1/2], [2,14,-d/(d - 1)], [2,36,-1/2],
        [2,37,1], [3,8,1], [4,9,-d/(d - 1)], [4,10,-1/(d - 1)],
        [4,14,d/(d - 1)], [5,9,-1/(d - 1)], [5,10,-1/(d - 1)], [5,14,1/(d - 1)],
        [6,4,d + 1], [6,5,-d], [6,8,-d], [6,9,d/(d - 1)],
        [6,10,d/(d - 1)], [6,11,3*d], [6,12,-2*d], [6,13,d/2],
        [6,14,-d^2/(d - 1)], [6,36,-d/2], [6,37,d], [7,2,1],
        [7,3,-1], [8,2,-d/(d - 1)], [8,4,-d^2/(d - 1)], [8,5,d],
        [8,6,d/(d - 1)], [8,8,d], [8,9,-d/(d - 1)], [8,10,-d/(d - 1)],
        [8,11,-3*d], [8,12,2*d], [8,13,-d/2], [8,14,d^2/(d - 1)],
        [8,36,d/2], [8,37,-d], [9,2,-d/(d - 1)], [9,4,-d/(d - 1)],
        [9,5,1], [9,6,1/(d - 1)], [9,8,1], [9,9,-1/(d - 1)],
        [9,10,-1/(d - 1)], [9,11,-3], [9,12,2], [9,13,-1/2],
        [9,14,d/(d - 1)], [9,36,1/2], [9,37,-1], [10,1,-1],
        [11,1,2*d], [11,2,d*(3*d - 1)/(2*(d - 1))], [11,3,-d/2], [11,4,d^2/(d - 1)],
        [11,5,-d], [11,6,-d/(d - 1)], [11,8,-d], [11,9,d/(d - 1)],
        [11,10,d/(d - 1)], [11,14,-d/(d - 1)], [11,26,1], [11,35,-d],
        [11,37,d], [12,1,2], [12,2,(3*d - 1)/(2*(d - 1))], [12,3,-1/2],
        [12,4,d/(d - 1)], [12,5,-1], [12,6,-1/(d - 1)], [12,8,-1],
        [12,9,1/(d - 1)], [12,10,1/(d - 1)], [12,14,-1/(d - 1)], [12,35,-1],
        [12,37,1], [13,17,-1], [13,18,1], [14,7,-d],
        [14,11,-d/(d - 1)], [14,15,-d*(3*d - 2)/(d - 1)], [14,16,2*d], [14,17,-1/(d - 1)],
        [14,19,d/(d - 1)], [14,20,2*d/(d - 1)], [14,21,-2*d/(d - 1)], [14,22,3*d],
        [14,23,-2*d], [14,24,d], [14,25,-2*d^2/(d - 1)], [14,28,1],
        [14,32,d/(d - 1)], [14,38,d], [14,40,-d], [15,7,-1],
        [15,11,-1/(d - 1)], [15,15,-(3*d - 2)/(d - 1)], [15,16,2], [15,17,-1/(d - 1)],
        [15,19,1/(d - 1)], [15,20,2/(d - 1)], [15,21,-2/(d - 1)], [15,22,3],
        [15,23,-2], [15,24,1], [15,25,-2*d/(d - 1)], [15,28,1/d],
        [15,32,1/(d - 1)], [15,38,1], [15,40,-1], [16,34,-1],
        [17,1,-2], [17,2,-(3*d - 1)/(2*(d - 1))], [17,3,1/2], [17,4,-d/(d - 1)],
        [17,5,1], [17,6,1/(d - 1)], [17,8,1], [17,9,-1/(d - 1)],
        [17,10,-1/(d - 1)], [17,14,1/(d - 1)], [17,26,-1/(d - 1)], [17,27,1/(d - 1)],
        [17,31,1/(d - 1)], [17,33,d/(d - 1)], [17,35,1], [17,37,-1],
        [18,1,-2*d], [18,2,-d*(3*d - 1)/(2*(d - 1))], [18,3,d/2], [18,4,-d^2/(d - 1)],
        [18,5,d], [18,6,d/(d - 1)], [18,8,d], [18,9,-d/(d - 1)],
        [18,10,-d/(d - 1)], [18,14,d/(d - 1)], [18,26,-d/(d - 1)], [18,27,d/(d - 1)],
        [18,31,d/(d - 1)], [18,33,d/(d - 1)], [18,35,d], [18,37,-d],
        [19,11,-1], [19,12,1], [20,14,1], [21,13,1],
        [22,11,-1], [25,15,-1], [25,16,1], [26,7,d],
        [26,15,3*d], [26,16,-2*d], [26,20,-2*d/(d - 1)], [26,21,2*d/(d - 1)],
        [26,22,-3*d], [26,23,2*d], [26,24,-d], [26,25,2*d^2/(d - 1)],
        [26,28,-1], [26,32,-d/(d - 1)], [26,38,-d], [26,40,d],
        [27,7,1], [27,15,2], [27,16,-2], [27,20,-2/(d - 1)],
        [27,21,2/(d - 1)], [27,22,-3], [27,23,2], [27,24,-1],
        [27,25,2*d/(d - 1)], [27,28,-1/d], [27,32,-1/(d - 1)], [27,38,-1],
        [27,40,1], [28,7,-1], [29,20,-d/(d - 1)], [29,21,1/(d - 1)],
        [29,22,-3*d/2], [29,23,d], [29,24,-d/2], [29,25,d^2/(d - 1)],
        [29,28,-3*d/2], [29,29,d], [29,30,-d/2], [29,31,d],
        [29,32,-d/(2*(d - 1))], [29,39,d/2], [29,40,d/2], [30,20,-1/(d - 1)],
        [30,21,1/(d - 1)], [30,22,-3/2], [30,23,1], [30,24,-1/2],
        [30,25,d/(d - 1)], [30,28,-3/2], [30,29,1], [30,30,-1/2],
        [30,31,1], [30,32,-1/(2*(d - 1))], [30,39,1/2], [30,40,1/2],
        [31,22,-1], [31,23,1], [32,22,1], [32,28,1],
        [32,32,d/(d - 1)], [33,22,1], [33,28,1/d], [33,32,1/(d - 1)],
        [34,24,-1], [35,22,3/2], [35,23,-1], [35,24,1/2],
        [35,28,3/2], [35,29,-1], [35,30,1/2], [35,31,-1],
        [35,32,1/(2*(d - 1))], [35,39,-1/2], [35,40,-1/2], [36,22,3*d/2],
        [36,23,-d], [36,24,d/2], [36,25,-d], [36,28,3*d/2],
        [36,29,-d], [36,30,d/2], [36,31,-d], [36,32,d/(2*(d - 1))],
        [36,39,-d/2], [36,40,-d/2], [37,28,-1], [37,29,1],
        [38,31,1], [39,30,-1], [40,28,(d - 1)/d]]:
end proc:

LITAODD_CYCLEU:=proc(d)
    return [[6,1,1], [6,2,-d], [6,4,1/d], [6,5,1/d],
        [6,6,1], [14,10,-1], [14,12,-d], [14,14,-1],
        [14,16,-1], [14,22,1], [24,10,1], [24,12,d - 1],
        [24,22,-1], [8,1,-1], [8,2,d - 1], [5,4,1],
        [5,7,-1], [5,8,-1], [5,23,1], [13,14,-1],
        [13,15,-1], [13,17,-1], [13,20,-1], [13,21,1],
        [7,2,-1], [7,3,-1], [7,17,-1], [7,18,1],
        [19,7,1], [19,11,d], [19,12,-1], [19,13,-1],
        [19,22,1], [15,10,1], [15,12,d], [15,14,d],
        [15,16,1], [15,19,1], [15,21,1], [15,22,-1],
        [2,1,-1], [2,2,d], [2,4,-1], [2,5,-1],
        [2,6,-1], [2,9,-1], [2,23,1]]:
end proc:

LITAODD_CYCLEV:=proc(d)
    return [[4,7,1], [4,8,1], [4,10,1], [4,11,1],
        [4,22,1], [11,17,1], [11,18,1], [11,19,1],
        [11,20,1], [11,21,1], [18,19,-1], [18,20,-(d - 1)/d],
        [18,21,-1], [23,10,-1], [23,11,-(d - 1)/d], [23,22,-1],
        [6,4,1], [6,5,1], [6,6,1], [6,23,1],
        [7,2,1], [7,3,1], [7,6,1], [13,14,1],
        [13,15,1], [13,16,1], [13,20,1], [13,21,1],
        [19,11,-1/d], [19,12,1], [19,13,1], [19,16,1],
        [19,22,-1], [10,1,1], [10,17,-1], [10,18,-d],
        [10,19,-1], [10,20,-1], [10,21,-1], [5,7,-1],
        [5,8,-d], [5,9,1], [5,10,-1], [5,11,-1],
        [5,22,-1], [5,23,1]]:
end proc:

LITAODD_CYCLEW:=proc(d)
    return [[3,9,-1], [16,19,-1], [10,1,-1], [21,10,-1],
        [5,7,-1/(d - 1)], [5,8,1/(d - 1)], [5,10,-1], [5,11,1/(d - 1)],
        [5,12,1/(d - 1)], [5,13,(d - 3)/(d - 1)], [5,14,1/(d - 1)], [5,15,-1/(d - 1)],
        [5,16,-1/(d - 1)], [5,19,1], [5,20,-1], [5,21,-1],
        [5,22,-1], [9,2,1/(d - 1)], [9,3,-1/(d - 1)], [9,5,-1/(d - 1)],
        [9,6,1/(d - 1)], [15,4,1], [15,5,-2], [15,7,-1/(d - 1)],
        [15,8,1/(d - 1)], [15,9,-1], [15,10,-1], [15,11,1/(d - 1)],
        [15,12,1/(d - 1)], [15,13,(d - 3)/(d - 1)], [15,14,1/(d - 1)], [15,15,-1/(d - 1)],
        [15,16,-1/(d - 1)], [15,22,-1], [15,23,-1], [12,15,1/(d - 1)],
        [12,17,1/(d - 1)], [12,18,1/(d - 1)], [12,20,-1/(d - 1)], [17,15,1],
        [17,20,-1], [2,5,-1], [20,10,1], [20,12,-1/(d - 1)],
        [20,13,-(d - 2)/(d - 1)], [20,14,-1/(d - 1)], [20,15,1/(d - 1)], [20,16,1/(d - 1)],
        [20,19,-1], [20,20,1], [20,21,1], [20,22,1],
        [22,4,-1], [22,5,2], [22,7,1/(d - 1)], [22,8,-1/(d - 1)],
        [22,9,1], [22,10,1], [22,11,-1/(d - 1)], [22,13,-(d - 2)/(d - 1)],
        [22,22,1], [22,23,1], [7,3,-1], [13,15,-1],
        [1,4,-1], [1,5,1], [19,13,-1], [23,10,d],
        [23,11,-1], [23,12,-d/(d - 1)], [23,13,-(d^2 - 3*d + 1)/(d - 1)], [23,14,-d/(d - 1)],
        [23,15,d/(d - 1)], [23,16,d/(d - 1)], [23,19,-d], [23,20,d],
        [23,21,d], [23,22,d], [11,15,d/(d - 1)], [11,17,d/(d - 1)],
        [11,18,1/(d - 1)], [11,20,-d/(d - 1)], [8,2,1/(d - 1)], [8,3,-1/(d - 1)],
        [8,5,-d/(d - 1)], [8,6,d/(d - 1)], [24,4,-d], [24,5,2*d],
        [24,7,d/(d - 1)], [24,8,-d/(d - 1)], [24,9,d], [24,10,d],
        [24,11,-d/(d - 1)], [24,12,-1], [24,13,-(d^2 - 3*d + 1)/(d - 1)], [24,22,d],
        [24,23,d], [4,7,-d/(d - 1)], [4,8,1/(d - 1)], [4,10,-d],
        [4,11,d/(d - 1)], [4,12,d/(d - 1)], [4,13,d*(d - 3)/(d - 1)], [4,14,d/(d - 1)],
        [4,15,-d/(d - 1)], [4,16,-d/(d - 1)], [4,19,d], [4,20,-d],
        [4,21,-d], [4,22,-d], [14,4,d], [14,5,-2*d],
        [14,7,-d/(d - 1)], [14,8,d/(d - 1)], [14,9,-d], [14,10,-d],
        [14,11,d/(d - 1)], [14,12,d/(d - 1)], [14,13,d*(d - 3)/(d - 1)], [14,14,1/(d - 1)],
        [14,15,-1/(d - 1)], [14,16,-d/(d - 1)], [14,22,-d], [14,23,-d]]:
end proc:


LITAODD_BASE:=proc(A,B,C,m,d)
local expr,call,i,j,k,q:
    expr:=0:

    for i from 0 to m-1 do
        expr:=expr+LITAODD_LABEL_TERM(A,B,C,[s0,0,i],m,d):
        expr:=expr+LITAODD_LABEL_TERM(A,B,C,[s0,1,i],m,d):
    end do:

    for call from 0 to 1 do
        for i from 0 to d-1 do
            for j from 0 to d-1 do
                for k from 0 to d-1 do
                    if ((i<=j and j<k) or (k<j and j<=i))
                        and LITAODD_PROPSIZE(i,j,k,d) then
                        expr:=expr+LITAODD_LABEL_TERM(A,B,C,[s1,call,i,j,k],m,d):
                    end if:
                end do:
            end do:
        end do:
    end do:

    for call from 0 to 1 do
        for i from 0 to d-1 do
            for j from 0 to d-1 do
                for k from 0 to d-1 do
                    if not (i=j and j=k) and
                        LITAODD_PROPSIZE(i,j,k,d) then
                        expr:=expr+LITAODD_LABEL_TERM(A,B,C,[s2,call,i,j,k],m,d):
                    end if:
                end do:
            end do:
        end do:
    end do:

    for i from 0 to m-1 do
        expr:=expr+LITAODD_LABEL_TERM(A,B,C,[u2p,0,i],m,d):
        expr:=expr+LITAODD_LABEL_TERM(A,B,C,[u2p,1,i],m,d):
    end do:

    for q from 0 to 5 do
        for i from 0 to m-1 do
            expr:=expr+LITAODD_LABEL_TERM(A,B,C,[u2,q,i],m,d):
        end do:
    end do:

    for q from 0 to 7 do
        for i from 0 to m-1 do
            for j from 0 to m-1 do
                if i<>j then
                    expr:=expr+LITAODD_LABEL_TERM(A,B,C,[u1,q,i,j],m,d):
                end if:
            end do:
        end do:
    end do:

    for q from 0 to 3 do
        for i from 0 to m-1 do
            expr:=expr+LITAODD_LABEL_TERM(A,B,C,[u3,q,i],m,d):
        end do:
        for j from 0 to m-1 do
            expr:=expr+LITAODD_LABEL_TERM(A,B,C,[u4,q,j],m,d):
        end do:
    end do:

    return expr:
end proc:

LITAODD_LOCAL_REPLACEMENTS:=proc(A,B,C,m,d)
local blocks,block,groups,expr,p4u,p4v,p4w,p3u,p3v,p3w,cu,cv,cw,z,t:
    z:=m-1:
    p4u:=LITAODD_PATH4U(d): p4v:=LITAODD_PATH4V(d): p4w:=LITAODD_PATH4W(d):
    p3u:=LITAODD_PATH3U(d): p3v:=LITAODD_PATH3V(d): p3w:=LITAODD_PATH3W(d):
    cu:=LITAODD_CYCLEU(d): cv:=LITAODD_CYCLEV(d): cw:=LITAODD_CYCLEW(d):
    blocks:=LITAODD_INDEX_BLOCKS(m):
    expr:=0:

    for t from 1 to nops(blocks) do
        block:=blocks[t]:
        if nops(block)=4 then
            groups:=LITAODD_ORDERED_PATH4_GROUPS(block,z):
            expr:=expr-LITAODD_REMOVE_SUM(A,B,C,groups,m,d)
                +LITAODD_LOCAL_SUM(A,B,C,groups,p4u,p4v,p4w,57,m,d):
        elif nops(block)=3 then
            groups:=LITAODD_ORDERED_PATH3_GROUPS(block,z):
            expr:=expr-LITAODD_REMOVE_SUM(A,B,C,groups,m,d)
                +LITAODD_LOCAL_SUM(A,B,C,groups,p3u,p3v,p3w,40,m,d):
        elif nops(block)=2 then
            groups:=LITAODD_ORDERED_CYCLE_GROUPS(block[1],block[2],z):
            expr:=expr-LITAODD_REMOVE_SUM(A,B,C,groups,m,d)
                +LITAODD_LOCAL_SUM(A,B,C,groups,cu,cv,cw,23,m,d):
        elif nops(block)<>1 then
            error "unsupported local block"
        end if:
    end do:

    return expr:
end proc:

LITAODD:=proc(triad :: TRIAD)

description
    "Given a triad, return the trilinear form of the odd-dimensional LITA",
    "local improvement to trilinear aggregation",
    COPYLEFT :

local
    dims,m,d,A,B,C
:
    dims := TRIADDIM(triad) :

    if nops(convert(dims,`set`))<>1
    then error "triad should be square"
    end if:

    if map(irem,dims,2)<>[1,1,1]
    then error "triad's dimension should be odd"
    end if:

    if dims[1] < 19
    then error "triad's dimension should be at least 19"
    end if:

    m:=iquo(dims[1]+1,2):
    d:=m+1:
    A:=EXTRACTMAT(1,triad):
    B:=EXTRACTMAT(2,triad):
    C:=EXTRACTMAT(3,triad):

    return LITAODD_BASE(A,B,C,m,d)+LITAODD_LOCAL_REPLACEMENTS(A,B,C,m,d):
end proc:
