#Dynamic Programming:

#70. Climbing Stairs
# Time and Space O(n) for all methods

#recursive
class Solution:
    def climbStairs(self, n: int) -> int:
        res=[0]*(n+1)
        return self.helper(n,res)

    def helper(self,n,res):
        if(n<=1): return 1
        if res[n]>0: return res[n]
        res[n]=self.helper(n-1,res)+self.helper(n-2,res)
        return res[n]
#iterative       
class Solution:
    def climbStairs(self, n: int) -> int:
        dp=[0]*(n+1)
        dp[0],dp[1]=1,1
        for i in range(2,n+1):
            dp[i]=dp[i-1]+dp[i-2]
        return dp[n]
  
#746. Min Cost Climbing Stairs
# Time and Space O(n) for all methods
# Or space O(1)

#Itrative,space O(1)    
class Solution:
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        res=0
        dp1,dp2=0,0
        for i in range(2,len(cost)+1):
            dp=min(dp1+cost[i-1],dp2+cost[i-2])
            dp2=dp1
            dp1=dp
        return dp
    
#Itrative,space O(n)    
class Solution:
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        res=0
        dp=[0]*(len(cost)+1)
        dp[0],dp[1]=0,0
        for i in range(2,len(cost)+1):
            dp[i]=min(dp[i-1]+cost[i-1],dp[i-2]+cost[i-2])
        return dp[len(cost)]        
       
#303. Range Sum Query - Immutable   
class NumArray:
    def __init__(self, nums: List[int]):
        self.dp = nums
        for i in range(1,len(nums)):
            self.dp[i]+=self.dp[i-1]
    def sumRange(self, i: int, j: int) -> int:
        if i==0:
            return self.dp[j]
        else:
            return self.dp[j]-self.dp[i-1]
        
#53. Maximum Subarray    
#Time O(n), space O(1)
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        cursum=0
        res=float('-Inf')
        for i in range(len(nums)):
            cursum = max(cursum + nums[i], nums[i])
            res = max(res, cursum)
        return res

#121. Best Time to Buy and Sell Stock
#Time O(n),space O(1)
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        buy=float('Inf')
        res=0
        for i in range(len(prices)):
            buy=min(prices[i],buy)
            res=max(res,prices[i]-buy)
        return res
    
#Divide and Conquer
#169. Majority Element

#Moore Voting time:O(n) space O(1)
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        res=0
        cnt=0
        for num in nums:
            if cnt==0:
                res=num
                cnt+=1
            else:
                if num==res:
                    cnt+=1
                else:
                    cnt-=1
        return res
        
#2. Add Two Numbers
# time:O(max(m,n)), soace:O(max(m,n))+1 where m and n are length for l1,l2
class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        dummy=ListNode(-1)
        cur=dummy
        carry=0
        while l1 !=None or l2 !=None:
            d1 = l1.val if l1 else 0
            d2 = l2.val if l2 else 0
            res=d1+d2+carry
            carry = 1 if res>=10 else 0
            cur.next=ListNode(res%10)
            cur=cur.next
            l1 = l1.next if l1 else None
            l2 = l2.next if l2 else None
        if carry==1:
            cur.next=ListNode(1)
        return dummy.next

#445. Add Two Numbers II   
time:O(max(m,n)), soace:O(max(m,n))+1+m+n where m and n are length for l1,l2
class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        s1=[]
        s2=[]
        while l1!=None:
            s1.append(l1.val)
            l1=l1.next
        while l2!=None:
            s2.append(l2.val)
            l2=l2.next
        cur=ListNode((s1[-1]+s2[-1])%10)
        carry=1 if s1[-1]+s2[-1]>9 else 0
        s1.pop()
        s2.pop()
        while  s1 or  s2:
            d1=s1[-1] if s1 else 0
            d2=s2[-1] if s2 else 0
            sum=d1+d2+carry
            carry=1 if sum>9 else 0
            temp=ListNode(sum%10)
            temp.next=cur
            cur=temp
            s1.pop() if s1 else []
            s2.pop() if s2 else []
        if carry==1:
            temp=ListNode(1)
            temp.next=cur
            cur=temp
        return cur

#141. Linked List Cycle
#Time:O(n), space:O(1)
class Solution(object):
    def hasCycle(self, head):
        fast,slow=head,head
        while fast!=None and fast.next!=None:
            fast=fast.next.next
            slow=slow.next
            if fast==slow:
                return True
        return False
    
#142. Linked List Cycle II
#Time:O(n), space:O(1)
class Solution(object):
    def detectCycle(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        fast,slow=head,head
        while fast and fast.next:
        #while fast !=None and fast.next!=None: 
            fast=fast.next.next
            slow=slow.next
            if slow==fast:
                break
        if not fast or not fast.next: return None
        #if fast== None or fast.next == None: return None
        slow=head
        while slow!=fast:
            slow=slow.next
            fast=fast.next
        return slow

#Graph    
#133. Clone Graph
#DFS Solution, both time and space are O(n)
"""
# Definition for a Node.
class Node:
    def __init__(self, val, neighbors):
        self.val = val
        self.neighbors = neighbors
"""
class Solution:
    def cloneGraph(self, node: 'Node') -> 'Node':
        dict={}
        return self.helper(node,dict)
    def helper(self,node,dict):
        if node ==None: return None
        if node in dict:
            return dict[node]
        dup=Node(node.val,[])
        dict[node]=dup
        for neighbor in node.neighbors:
            clone=self.helper(neighbor,dict)
            dup.neighbors.append(clone)
        return dup
#BFS Solution, both time and space are O(n)
class Solution:
    def cloneGraph(self, node: 'Node') -> 'Node':
        if not node: return None
        dict={}
        queue=[node]
        dup=Node(node.val,[])
        dict[node]=dup
        while queue:
            t=queue.pop(0)
            for neighbor in t.neighbors:
                if neighbor not in dict:
                    dict[neighbor]=Node(neighbor.val,[])
                    queue.append(neighbor)
                dict[t].neighbors.append(dict[neighbor])
        return dup

#138. Copy List with Random Pointer
#Time and space O(n)
"""
# Definition for a Node.
class Node:
    def __init__(self, val, next, random):
        self.val = val
        self.next = next
        self.random = random
"""
class Solution:
    def copyRandomList(self, head: 'Node') -> 'Node':
        if head == None: return None
        dict={}
        node=head
        while node:
            dict[node]=Node(node.val,None,None)
            node=node.next
        node=head
        while node:
            if node.next:
                dict[node].next=dict[node.next]
            if node.random:
                dict[node].random=dict[node.random]
            node=node.next
        return dict[head]
    
#200. Number of Islands    
#Time and space O(m*n)
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        m=len(grid)
        if m==0 : return 0
        n=len(grid[0])
        visited=[[0]*n for _ in range(m) ]
        res=0
        for i in range(m):
            for j in range(n):
                if grid[i][j]=='1' and visited[i][j]==0:
                    self.helper(grid,visited,i,j)
                    res+=1
        return res
    
    def helper(self,grid,visited,x,y):
        m,n=len(grid),len(grid[0])
        if x<0 or x>=m or y<0 or y>=n or visited[x][y]==1 or grid[x][y] !='1': return
        visited[x][y]=1
        self.helper(grid,visited,x-1,y)
        self.helper(grid,visited,x+1,y)
        self.helper(grid,visited,x,y+1)
        self.helper(grid,visited,x,y-1)
 
#547. Friend Circles
#Time O(n^2),space=O(1)
class Solution:
    def findCircleNum(self, M: List[List[int]]) -> int:
        if not M:return 0
        m=len(M)
        visited=[0]*m
        res=0
        for i in range(m):
            if visited[i]==1: continue
            self.helper(M,i,m,visited)
            res+=1
        return res
    
    def helper(self,M,i,m,visited):
        if visited[i]==1:return
        visited[i]=1
        for j in range(m):
            if M[i][j] and visited[j]==0:
                self.helper(M,j,m,visited)


#695. Max Area of Island
#Time and space O(m*n)
class Solution:
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        m=len(grid)
        if m==0 : return 0
        n=len(grid[0])
        res=0
        for i in range(m):
            for j in range(n):
                if grid[i][j]==1:
                    area=self.helper(grid,i,j,0)
                    res=max(res,area)   
        return res
    
    def helper(self,grid,x,y,area):
        m,n=len(grid),len(grid[0])
        if x<0 or x>=m or y<0 or y>=n or grid[x][y]==0: return area
        grid[x][y]=0
        area+=1
        area=self.helper(grid,x-1,y,area)
        area=self.helper(grid,x+1,y,area)
        area=self.helper(grid,x,y-1,area)
        area=self.helper(grid,x,y+1,area)
        return area

#733. Flood Fill
#Time=O(m*n),space=O(1)
class Solution:
    def floodFill(self, image: List[List[int]], sr: int, sc: int, newColor: int) -> List[List[int]]:
        if image[sr][sc]==newColor: return image
        m,n=len(image),len(image[0])
        self.helper(image,sr,sc,image[sr][sc],newColor)
        return image
    def helper(self,image,x,y,preColor,newColor):
        m,n=len(image),len(image[0])
        if x<0 or x>=m or y<0 or y>=n: return
        if image[x][y] !=preColor: return
        image[x][y]=newColor
        self.helper(image,x+1,y,preColor,newColor)
        self.helper(image,x-1,y,preColor,newColor)
        self.helper(image,x,y+1,preColor,newColor)
        self.helper(image,x,y-1,preColor,newColor)

    
    #Binary Seaarch
    #34. Find First and Last Position of Element in Sorted Array
    #One binary search, time O(n) in stead of O(logn) when whole array can have same number, space=O(1)
    class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        idx=self.helper(nums,0,len(nums)-1,target)
        if idx==-1: return[-1,-1]
        left,right=idx,idx
        while left>0 and nums[left-1]==nums[idx]:
            left-=1
        while right<len(nums)-1 and nums[right+1]==nums[idx]:
            right+=1
        return [left,right]
    
    def helper(self,nums,left,right,target):
        if left>right: return -1
        mid=left+int((right-left)/2)
        if target==nums[mid]: return mid
        elif nums[mid]<target:
            return self.helper(nums,mid+1,right,target)
        else: return self.helper(nums,left,mid-1,target)
       
#Two binary search, time=O(logn), space=O(1)
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        res=[-1]*2
        if len(nums)==0: return res
        left,right=0,len(nums)-1
        while(left<=right):
            mid=left+int((right-left)/2)
            if nums[mid]<target: left=mid+1
            else: right=mid-1
        if right+1==len(nums): return res
        if nums[right+1] !=target: return res
        res[0]=right+1
        right=len(nums)-1
        while left<=right:
            mid=left+int((right-left)/2)
            if nums[mid]<=target: left=mid+1
            else: right=mid-1
        res[1]=right
        return res

#35. Search Insert Position
#Time=O(logn), space=O(1)
class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        left,right=0,len(nums)-1
        while left<= right:
            mid=left+int((right-left)/2)
            if nums[mid]<target:
                left=mid+1
            else:
                right=mid-1
        return right+1
            

#704. Binary Search
#Time=O(logn), space=O(1)
class Solution:
    def search(self, nums: List[int], target: int) -> int: 
        left,right=0,len(nums)-1
        while left<=right:
            mid=left+int((right-left)/2)
            if nums[mid]==target:
                return mid
            elif nums[mid]>target:
                right=mid-1
            else:
                left=mid+1
        return -1
    
#33. Search in Rotated Sorted Array
#Time O(logn), space O(n)
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        left=0
        right=len(nums)-1
        while left<=right:
            mid=int(left+(right-left)/2)
            if target==nums[mid]:
                return mid
            elif nums[mid]<nums[right]:
                if nums[mid]<target and nums[right]>=target:
                    left=mid+1
                else: right=mid-1
            else:
                if nums[left]<=target and nums[mid]> target:
                    right=mid-1
                else: left=mid+1
        return -1
 
#Tree:
#94. Binary Tree Inorder Traversal
#Time=O(n), space=O(n) for all methods.
#Recursive solution
class Solution:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        res=[]
        if(not root): return res
        self.helper(root,res)
        return res
    def helper(self,root,res):
        if(not root): return
        self.helper(root.left,res)
        res.append(root.val)
        self.helper(root.right,res)
        
#Using stack
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        res=[]
        if(root==None): return res
        s=[]
        cur=root
        while cur or s:
            while cur:
                s.append(cur)
                cur=cur.left
            cur=s.pop()
            res.append(cur.val)
            cur=cur.right
        return res
#144. Binary Tree Preorder Traversal
#Time O(n), space=O(n) for both methods
#Iterative:    
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def preorderTraversal(self, root: TreeNode) -> List[int]:
        s=[root]
        res=[]
        if root==None:
            return res
        while s :
            cur=s.pop()
            res.append(cur.val)
            if cur.right != None: s.append(cur.right)
            if cur.left != None: s.append(cur.left)
        return res
#Recursive:
class Solution:
    def preorderTraversal(self, root: TreeNode) -> List[int]:     
        res=[]
        if root==None: return res
        self.helper(root,res)
        return res
    def helper(self,root,res):
        if root==None: return
        res.append(root.val)
        self.helper(root.left,res)
        self.helper(root.right,res)
            

#589. N-ary Tree Preorder Traversal
#Time O(n), space=O(n) for both methods
#Iterative
class Solution:
def preorder(self, root: 'Node') -> List[int]:
    res=[]
    if root==None: return res
    s=[root]
    while s:
        cur=s.pop()
        res.append(cur.val)
        for i in range(len(cur.children)-1,-1,-1):
            s.append(cur.children[i])
    return res
#Recursive
class Solution:
def preorder(self, root: 'Node') -> List[int]:
    res=[]
    if root==None: return res
    self.helper(root,res)
    return res
def helper(self,root,res):
    if root==None: return 
    res.append(root.val)
    for child in root.children:
        self.helper(child,res)

#145. Binary Tree Postorder Traversal
#Time=O(n), space=O(n)
#Iterative
class Solution:
    def postorderTraversal(self, root: TreeNode) -> List[int]:
        res=[]
        if root==None: return res
        s=[root]
        while s:
            cur=s.pop()
            res.append(cur.val)
            if cur.left:
                s.append(cur.left)
            if cur.right:
                s.append(cur.right)
        return res[::-1]
#Recursive   
class Solution:
    def postorderTraversal(self, root: TreeNode) -> List[int]:
        res=[]
        if root==None: return res
        self.helper(root,res)
        return res
    def helper(self,root,res):
        if root==None: return 
        self.helper(root.left,res)
        self.helper(root.right,res)
        res.append(root.val)
        
#590. N-ary Tree Postorder Traversal
#Time=O(n), space=O(n)
class Solution:
    def postorder(self, root: 'Node') -> List[int]:
        res=[]
        if root==None:return res
        s=[root]
        while s:
            cur=s.pop()
            res.append(cur.val)
            for child in cur.children:
                s.append(child)
        return res[::-1]
                

#114. Flatten Binary Tree to Linked List
#Time=O(n), spcae=O(1)
class Solution:
    def flatten(self, root: TreeNode) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        self.pre = None
        self.helper(root)

    def helper(self,root):
        if not root:return None
        self.helper(root.right)
        self.helper(root.left)
        root.right = self.pre
        root.left = None
        self.pre = root

#102. Binary Tree Level Order Traversal        
#Time=O(n), space=O(n) for both BFS and DFS solutions
#BFS solution
class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        res=[]
        if root == None:
            return res
        q=[root]
        while q:
            res_sub=[]
            for i in range(len(q)):
                t=q.pop(0)
                res_sub.append(t.val)
                if t.left !=None: 
                    q.append(t.left)
                if t.right !=None:
                    q.append(t.right)
            res.append(res_sub)
        return res
        
#DFS solution:
class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        res=[]
        self.helper(root,0,res)
        return res
    def helper(self,root,depth,res):
        if root==None: return
        while len(res)<=depth:
            res.append([])
        res[depth].append(root.val)
        self.helper(root.left,depth+1,res)
        self.helper(root.right,depth+1,res)
        
#107. Binary Tree Level Order Traversal II
#Time=O(n), space=O(n) for both BFS and DFS solutions
#BFS solution
class Solution:
    def levelOrderBottom(self, root: TreeNode) -> List[List[int]]:
        res=[]
        if not root: return  res
        q=[root]
        while q:
            res_sub=[]
            for i in range(len(q)):
                t=q.pop(0)
                res_sub.append(t.val)
                if t.left !=None: 
                    q.append(t.left)
                if t.right !=None:
                    q.append(t.right)
            res.append(res_sub)
        return res[::-1]

#DFS, preorder
class Solution:
    def levelOrderBottom(self, root: TreeNode) -> List[List[int]]:
        res=[]
        self.helper(root,0,res)
        return res[::-1]
    def helper(self,root,depth,res):
        if root==None: return
        while len(res)<=depth:
            res.append([])
        res[depth].append(root.val)
        self.helper(root.left,depth+1,res)
        self.helper(root.right,depth+1,res)
 
#108. Convert Sorted Array to Binary Search Tree
#Time=O(nlogn),space=O(n)
class Solution:
    def sortedArrayToBST(self, nums: List[int]) -> TreeNode:
        return self.helper(nums,0,len(nums)-1)
    def helper(self,nums,left,right):
        if left>right: return None
        mid=left+int((right-left)/2)
        cur=TreeNode(nums[mid])
        cur.left=self.helper(nums,left,mid-1)
        cur.right=self.helper(nums,mid+1,right)
        return cur

#129. Sum Root to Leaf Numbers
#Time=O(n),space=O(1)
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def sumNumbers(self, root: TreeNode) -> int:
        return self.helper(root,0)
    def helper(self,root,prefix):
        if root==None: return 0
        if root.left ==None and root.right==None:
            return prefix*10+root.val  
        return self.helper(root.left,prefix*10+root.val)+self.helper(root.right,prefix*10+root.val)

#669. Trim a Binary Search Tree
#Time=O(n),space=0(1)
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def trimBST(self, root: TreeNode, L: int, R: int) -> TreeNode:
        if root==None: return None
        if L<=root.val and R>=root.val:
            root.left=self.trimBST(root.left,L,R)
            root.right=self.trimBST(root.right,L,R)
        elif root.val<L:
            root=self.trimBST(root.right,L,R)
        else:
            root=self.trimBST(root.left,L,R)
        return root
    
    
    
#Search:
#46. Permutations
#Time=O(n!), space=O(n!), in fact recusion has the form T(n)=T(n-1）+T（n-2）+.... has the time complexity O(2^(n)),here since 
#this line: if visited[i]==1: continue , it reduce the time complexity from O(2^n) to O(n!)
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        res=[]
        out=[]
        visited=[0]*len(nums)
        self.helper(nums,0,visited,out,res)
        return res
    
    def helper(self,nums,level,visited,out,res):
        if level==len(nums):
            res.append(out[:])
            return
        for i in range(len(nums)):
            if visited[i]==1: continue
            visited[i]=1
            out.append(nums[i])
            self.helper(nums,level+1,visited,out,res)
            out.pop()
            visited[i]=0
            
#Time=O(n!), space=O(n)
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        res=[]
        self.helper(nums,0,res)
        return res
    def helper(self,nums,start,res):
        if start==len(nums):
            res.append(nums[:])
            return
        for i in range(start,len(nums)):
            nums[start],nums[i]=nums[i],nums[start]
            self.helper(nums,start+1,res)
            nums[start],nums[i]=nums[i],nums[start]

#47. Permutations II          
#Time=O(n!), space=O(n!)
class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        res=[]
        out=[]
        visited=[0]*len(nums)
        nums.sort()
        self.helper(nums,0,visited,out,res)
        return res
    def helper(self,nums,level,visited,out,res):
        if level==len(nums):
            res.append(out[:])
            return
        for i in range(len(nums)):
            if visited[i]==1: continue
            if i>0 and nums[i]==nums[i-1] and visited[i-1]==0: continue
            visited[i]=1
            out.append(nums[i])
            self.helper(nums,level+1,visited,out,res)
            out.pop()
            visited[i]=0

#784. Letter Case Permutation
#Time=O(n* 2^l, l the number of letters in a string), space=O(n)+O(n*2^l)
class Solution:
    def letterCasePermutation(self, S: str) -> List[str]:
        ans = []
        self.helper(list(S), 0, len(S),ans)
        return ans
    def helper(self,S, i, n,ans):
        if i == n:
            ans.append(''.join(S))
            return
        self.helper(S, i + 1, n,ans)      
        if not S[i].isalpha(): return      
        S[i] = chr(ord(S[i]) ^ (1<<5))
        self.helper(S, i + 1, n,ans)
        S[i] = chr(ord(S[i]) ^ (1<<5))

#22. Generate Parentheses
#Time=O(n!)， seems like O(2^n) but using Catalan number theory, get O(n!), space=O(n)
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        res=[]
        self.helper(n,n,'',res)
        return res
    def helper(self,left,right,out,res):
        if left<0 or right<0 or left>right: return
        if left==0 and right==0:
            res.append(out)
            return
        self.helper(left-1,right,out+'(',res)
        self.helper(left,right-1,out+')',res)

#72. Edit Distance
#Time=O(m*n),space=O(m*n)
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        n1=len(word1)+1
        n2=len(word2)+1
        dp=[[-1]*n2 for i in range(n1)]
        for i in range(n1):
            dp[i][0]=i
        for i in range(n2):
            dp[0][i]=i
        for i in range(1,n1):
            for j in range(1,n2):
                if word1[i-1]==word2[j-1]:
                    dp[i][j]=dp[i-1][j-1]
                else:
                    dp[i][j]=min(dp[i][j-1],dp[i-1][j],dp[i-1][j-1])+1
        return dp[n1-1][n2-1]

    
#207. Course Schedule
#Finding cycles O(n^2) -> Topological sort O(n), space=O(n)
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        graph = [[] for _ in range(numCourses)]
        for course, prerequisite in prerequisites:
            graph[prerequisite].append(course)
        visited=[0]*numCourses
        for i in range(numCourses):
            if self.helper(i,graph,visited):
                return False
        return True
    #If there is a cycle, return False
    def helper(self,curr,graph,visited):
        if visited[curr]==1: return True
        if visited[curr]==2: return False
        visited[curr]=1
        for ele in graph[curr]:
            if self.helper(ele,graph,visited):
                return True
        visited[curr]=2
        return False
    
#210. Course Schedule II
#Time=O(E+V)->O(n), space=O(n)
class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        graph = [[] for _ in range(numCourses)]
        res=[]
        for course, prerequisite in prerequisites:
            graph[prerequisite].append(course)
        visited=[0]*numCourses
        for i in range(numCourses):
            if self.helper(i,graph,visited,res):
                return []
        return res[::-1]
    #If there is a cycle, return False
    def helper(self,curr,graph,visited,res):
        if visited[curr]==1: return True
        if visited[curr]==2: return False
        visited[curr]=1
        for ele in graph[curr]:
            if self.helper(ele,graph,visited,res):
                return True
        visited[curr]=2
        res.append(curr)
        print(curr)
        return False
