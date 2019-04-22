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
