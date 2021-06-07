---


---

<h1 id="leetcode-101">Leetcode 101</h1>
<h2 id="第二章----最易懂的贪心算法">第二章 – 最易懂的贪心算法</h2>
<h4 id="assign-cookies">455. Assign Cookies</h4>
<pre><code>	public int findContentChildren(int[] g, int[] s) {
        Arrays.sort(g);
        Arrays.sort(s);
        int j = 0, i = 0;
        while(j &lt; s.length &amp;&amp; i &lt; g.length){
            if(s[j] &gt;= g[i]) i++;
            j++;
        }
        return i;
    }
</code></pre>
<h4 id="non-overlapping-intervals">435. Non-overlapping Intervals</h4>
<pre><code>    public int eraseOverlapIntervals(int[][] intervals) {
        if(intervals.length == 0) return 0;
        
        Arrays.sort(intervals, (a, b) -&gt; a[1] - b[1]);
        for(int[] i : intervals){
            System.out.println(i[0] + " " + i[1]);
        }
        int removed = 0, prev = intervals[0][1];
        for(int i = 1; i &lt; intervals.length; i++){
            if(intervals[i][0] &lt; prev){
                removed++;
            }
            else{
                prev = intervals[i][1];
            }
        }
        return removed;
    }
</code></pre>
<h2 id="第三章----玩转双指针">第三章 – 玩转双指针</h2>
<h4 id="two-sum-ii---input-array-is-sorted">167. Two Sum II - Input array is sorted</h4>
<pre><code>public int[] twoSum(int[] numbers, int target) {
    int l = 0, r = numbers.length - 1, sum;
    while(l &lt; r){
        sum = numbers[l] + numbers[r];
        if(sum == target) break;
        if(sum &lt; target) l++;
        else r--;
    }
    return new int[]{l + 1, r + 1};
}
</code></pre>
<h4 id="merge-sorted-array">88. Merge Sorted Array</h4>
<pre><code>public void merge(int[] nums1, int m, int[] nums2, int n) {
    int pos = m-- + n-- - 1;
    while(m &gt;= 0 &amp;&amp; n &gt;= 0){
        nums1[pos--] = nums1[m] &gt; nums2[n] ? nums1[m--] : nums2[n--];
    }
    while(n &gt;= 0){
        nums1[pos--] = nums2[n--];
    }
}
</code></pre>
<p>直观的写法：</p>
<pre><code>public void merge(int[] nums1, int m, int[] nums2, int n) {
    if(m == 0 &amp;&amp; n == 0) return;
    if(m != 0 &amp;&amp; n != 0){
        if(nums1[m - 1] &lt; nums2[n - 1]){
            nums1[m + n - 1] = nums2[n - 1];
            n--;
        }
        else{
            nums1[m + n - 1] = nums1[m - 1];
            m--;
        }
    }
    if(m != 0 &amp;&amp; n == 0){
        return;
    }
    if(m == 0 &amp;&amp; n != 0){
        nums1[m + n - 1] = nums2[n - 1];
        n--;
    }
    merge(nums1, m, nums2, n);
}
</code></pre>
<h4 id="linked-list-cycle-ii">142. Linked List Cycle II</h4>
<pre><code> public ListNode detectCycle(ListNode head) {
    ListNode slow = head;
    ListNode fast = head;
  
    do{
        if(fast == null || fast.next == null) return null;
        fast = fast.next.next;
        slow = slow.next;
    } while(fast != slow);
    
    fast = head;
    while(fast != slow){
        slow = slow.next;
        fast = fast.next;
    }
    return fast;
}
</code></pre>
<h4 id="minimum-window-substring">76. Minimum Window Substring</h4>
<pre><code>  public String minWindow(String s, String t) {
    int[] chars = new int[128];
    boolean[] flag = new boolean[128];
    
    //先统计t中的字符情况
    for(int i = 0; i &lt; t.length(); i++){
        flag[t.charAt(i)] = true;
        chars[t.charAt(i)]++;
    }
    
    //移动滑动窗口，不断更改统计数据
    int cnt = 0, l = 0, min_l = 0, min_size = s.length() + 1;
    for(int r = 0; r &lt; s.length(); r++){
        if(flag[s.charAt(r)]){
            chars[s.charAt(r)]--;
            if(chars[s.charAt(r)] &gt;= 0){
                cnt++;
            }
            
            //若目前滑动窗口已包含t中全部字符
            //则尝试将1右移，在不影响结果的情况下活的最短子字符串
            while(cnt == t.length()){
                if(r - l + 1 &lt; min_size){
                    min_l = l;
                    min_size = r - l + 1;
                }
                
                chars[s.charAt(l)]++;
                if(flag[s.charAt(l)] &amp;&amp; chars[s.charAt(l)] &gt; 0){
                    cnt--;
                }
                l++;
            }
        }
    }
    //substring() returns the substring from start index to end index
    return min_size &gt; s.length() ? "" : s.substring(min_l, min_l + min_size);
}
</code></pre>
<h2 id="第四章----居合斩！二分查找">第四章 – 居合斩！二分查找</h2>
<h4 id="sqrtx">69. Sqrt(x)</h4>
<pre><code>public int mySqrt(int x) {
    if(x == 0) return x;
    int l = 1, r = x, mid, sqrt;
    while(l &lt;= r){
        //why not use "mid = (l + r) / 2;" is explained  in the post
        //https://stackoverflow.com/questions/6735259/calculating-mid-in-binary-search
        mid = l + (r - l) / 2;
        sqrt = x / mid;
    
        if(sqrt == mid){
            return mid;
        }
        else if(mid &gt; sqrt){
            r = mid - 1;
        }
        else{
            l = mid + 1;
        }
    }
    return r;
}
</code></pre>
<p>牛顿迭代法：</p>
<pre><code>public int mySqrt(int x) {
    long num = x;
    while(num * num &gt; x){
        num = (num + x / num) / 2;
    }
    return (int) num;
}
</code></pre>
<h4 id="find-first-and-last-position-of-element-in-sorted-array">34. Find First and Last Position of Element in Sorted Array</h4>
<pre><code>public int[] searchRange(int[] nums, int target) {
    //左闭右开
    if(nums.length == 0) return new int[]{-1, -1};
    
    int lower = lower_bound(nums, target);
    int upper = upper_bound(nums, target) - 1;
    
    if(lower == nums.length || nums[lower] != target) return new int[]{-1, -1};
    
    return new int[]{lower, upper};
}

private int lower_bound(int[] nums, int target){
    int l = 0, r = nums.length, mid;
    while(l &lt; r){
        mid = l + (r - l) / 2;
        if(nums[mid] &gt;= target){
            r = mid;
        }
        else{
            l = mid + 1;
        }
    }
    return l;
}
private int upper_bound(int[] nums, int target){
    int l = 0, r = nums.length, mid;
    while(l &lt; r){
        mid = l + (r - l) / 2;
        if(nums[mid] &gt; target){
            r = mid;
        }
        else{
            l = mid + 1;
        }
    }
    return l;
}
</code></pre>
<h4 id="search-in-rotated-sorted-array-ii">81. Search in Rotated Sorted Array II</h4>
<pre><code>public boolean search(int[] nums, int target) {
    int l = 0, r = nums.length - 1, mid;
    
    //这里用的是双闭区间
    while(l &lt;= r){
        mid = l + (r - l) / 2;
        if(nums[mid] == target){
            return true;
        }
        if(nums[l] == nums[mid]){
            l++;
        }
        else if(nums[mid] &lt;= nums[r]){
            if(target &gt; nums[mid] &amp;&amp; target &lt;= nums[r]){
                l = mid + 1;
            }
            else{
                r = mid - 1;
            }
        }
        else{
            if(target &gt;= nums[l] &amp;&amp; target &lt; nums[mid]){
                r = mid -1;
            }
            else{
                l = mid + 1;
            }
        }   
    }
    return false;
}
</code></pre>
<h2 id="第五章----千奇百怪的排序算法">第五章 – 千奇百怪的排序算法</h2>
<h4 id="kth-largest-element-in-an-array">215. Kth Largest Element in an Array</h4>
<pre><code>public int findKthLargest(int[] nums, int k) {
    int l = 0, r = nums.length - 1, target = nums.length - k;
    while(l &lt; r){
        int mid = quickSelection(nums, l, r);
        if(mid == target){
            return nums[mid];
        }
        if(mid &lt; target){
            l = mid + 1;
        }
        else{
            r = mid - 1;
        }
    }
    return nums[l];
}

private int quickSelection(int[] nums, int l, int r){
    int i = l + 1, j = r, temp;
    while(true){
        while(i &lt; r &amp;&amp; nums[i] &lt;= nums[l]){
            i++;
        }
        while(l &lt; j &amp;&amp; nums[j] &gt;= nums[l]){
            j--;
        }
        if(i &gt;= j){
            break;
        }
        temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }
    temp = nums[l];
    nums[l] = nums[j];
    nums[j] = temp;
    return j;
}
</code></pre>
<p>下面这个是用了Hoare’s Partition的快选，是leetcode题解的第二个解法，写起来挺麻烦的，是教科书算法。 贴在这里方便对比参考。</p>
<pre><code>int [] nums;

public void swap(int a, int b) {
    int tmp = this.nums[a];
    this.nums[a] = this.nums[b];
    this.nums[b] = tmp;
}

public int partition(int left, int right, int pivot_index) {
    int pivot = this.nums[pivot_index];
    // 1. move pivot to end
    swap(pivot_index, right);
    int store_index = left;

    // 2. move all smaller elements to the left
    for (int i = left; i &lt;= right; i++) {
        if (this.nums[i] &lt; pivot) {
            swap(store_index, i);
            store_index++;        
        }
    }

    // 3. move pivot to its final place
    swap(store_index, right);

    return store_index;
}

public int quickselect(int left, int right, int k_smallest) {
/*
Returns the k-th smallest element of list within left..right.
*/

if (left == right) // If the list contains only one element,
  return this.nums[left];  // return that element

// select a random pivot_index
Random random_num = new Random();
int pivot_index = left + random_num.nextInt(right - left); 

pivot_index = partition(left, right, pivot_index);

// the pivot is on (N - k)th smallest position
if (k_smallest == pivot_index)
  return this.nums[k_smallest];
// go left side
else if (k_smallest &lt; pivot_index)
  return quickselect(left, pivot_index - 1, k_smallest);
// go right side
return quickselect(pivot_index + 1, right, k_smallest);
}

public int findKthLargest(int[] nums, int k) {
    this.nums = nums;
    int size = nums.length;
    // kth largest is (N - k)th smallest
    return quickselect(0, size - 1, size - k);
}
</code></pre>
<h4 id="top-k-frequent-elements">347. Top K Frequent Elements</h4>
<p>这里也可以把nums.length替换成C++版本的那个max_count(最高频率)，我试了一下，由于更慢我就没用了</p>
<pre><code>public int[] topKFrequent(int[] nums, int k) {
    Map&lt;Integer, Integer&gt; counts = new HashMap&lt;&gt;();
    for(int num : nums){
        counts.put(num, counts.getOrDefault(num, 0) + 1);
    }
    
    List&lt;Integer&gt;[] bucket = new List[nums.length + 1];
    
    for(int key : counts.keySet()){
        int i = counts.get(key);
        if(bucket[i] == null){
            bucket[i] = new ArrayList&lt;&gt;();
        }
        bucket[i].add(key);
    }
    
    List&lt;Integer&gt; list = new ArrayList&lt;&gt;();
    for(int i = nums.length; i &gt;= 0 &amp;&amp; list.size() &lt; k; i--){
        if(bucket[i] != null) list.addAll(bucket[i]);
    }
    
    int[] ret = new int[list.size()];
    int i = 0;
    for (Integer e : list) ret[i++] = e;
    return ret;
}
</code></pre>
<h2 id="第六章----一切皆可搜">第六章 – 一切皆可搜</h2>
<h4 id="max-area-of-island">695. Max Area of Island</h4>
<p>用stack实现dfs</p>
<pre><code>int[] direction = {-1, 0, 1, 0, -1};
public int maxAreaOfIsland(int[][] grid) {
    int m = grid.length, n = grid[0].length, local_area = 0, area = 0, x, y;
    for(int i = 0; i &lt; m; i++){
        for(int j = 0; j &lt; n; j++){
            if(grid[i][j] == 1){
                local_area = 1;
                grid[i][j] = 0;
                Stack&lt;Pair&lt;Integer, Integer&gt;&gt; island = new Stack();
                island.push(new Pair&lt;&gt;(i, j));
                while(!island.isEmpty()){
                    Pair&lt;Integer, Integer&gt; pair = island.pop();
                    for(int k = 0; k &lt; 4; k++){
                        x = pair.getKey() + direction[k];
                        y = pair.getValue() + direction[k + 1];
                        if(x &gt;= 0 &amp;&amp; x &lt; m &amp;&amp; y &gt;= 0 &amp;&amp; y &lt; n &amp;&amp; grid[x][y] == 1){
                           grid[x][y] = 0;
                           local_area++;
                           island.push(new Pair&lt;&gt;(x, y));
                        }
                    }     
                }
                area = Math.max(area, local_area);
            }
        }
    }
    return area;
}
</code></pre>
<p>用递归遍历四个方向，个人比较喜欢用递归写dfs的逻辑</p>
<pre><code>public int maxAreaOfIsland(int[][] grid) {
    int max_area = 0;
    for(int i = 0; i &lt; grid.length; i++){
        for(int j = 0; j &lt; grid[0].length; j++){
            if(grid[i][j] == 1){
                max_area = Math.max(max_area, dfs(grid, i, j));
            }
        }
    }
    return max_area;
}

private int dfs(int[][] grid, int r, int c){
    if(r &gt;= grid.length || r &lt; 0 || c &gt;= grid[0].length || c &lt; 0 ||
       grid[r][c] == 0){
        return 0;
    }
    grid[r][c] = 0;
    return dfs(grid, r + 1, c) + dfs(grid, r - 1, c) +
           dfs(grid, r, c + 1) + dfs(grid, r, c - 1) + 1;
}
</code></pre>
<h4 id="number-of-provinces">547. Number of Provinces</h4>
<p>这题名字变了，以前叫Friend Circles， 要注意。<br>
这题可以不设visited，直接把isConnected中的访问过的1变成0起同样效果。每次遍历从isConnected[i][i]开始</p>
<pre><code>public int findCircleNum(int[][] isConnected) {
    int count = 0;
    int[] visited = new int[isConnected.length];
    for(int i = 0; i &lt; isConnected.length; i++){
        if(visited[i] == 0){
            dfs(isConnected, visited, i);
            count++;
        }
    }
    return count;
}

private void dfs(int[][] isConnected, int[] visited, int i){
    visited[i] = 1;
    for(int j = 0; j &lt; isConnected[0].length; j++){
        if(isConnected[i][j] == 1 &amp;&amp; visited[j] == 0){
            dfs(isConnected, visited, j);
        }
    }
}
</code></pre>
<h4 id="pacific-atlantic-water-flow">417. Pacific Atlantic Water Flow</h4>
<pre><code>int[] direction = {-1, 0, 1, 0, -1};
public List&lt;List&lt;Integer&gt;&gt; pacificAtlantic(int[][] heights) {
    List&lt;List&lt;Integer&gt;&gt; ans = new LinkedList&lt;&gt;();
    
    if(heights == null || heights.length == 0 || heights[0].length == 0) return ans;
    
    int m = heights.length, n = heights[0].length;
    boolean[][] can_reach_p = new boolean[m][n];
    boolean[][] can_reach_a = new boolean[m][n];
    
    for(int i = 0; i &lt; m; i++){
        dfs(heights, can_reach_p, i, 0);
        dfs(heights, can_reach_a, i, n - 1);
    }
    
    for(int i = 0; i &lt; n; i++){
        dfs(heights, can_reach_p, 0, i);
        dfs(heights, can_reach_a, m - 1, i);
    }
    
    for(int i = 0; i &lt; m; i++){
        for(int j = 0; j &lt; n; j++){
         if(can_reach_p[i][j] &amp;&amp; can_reach_a[i][j]){
             //ans.add(List.of(i,j));
             ans.add(Arrays.asList(i, j));
         }   
        }
    }
    return ans;
}

private void dfs(int[][] heights, boolean[][] can_reach, int r, int c){
    if(can_reach[r][c]){
        return ;
    }
    can_reach[r][c] = true;
    int x, y;
    for(int i = 0; i &lt; 4; i++){
        x = r + direction[i];
        y = c + direction[i + 1];
        if(x &gt;= 0 &amp;&amp; x &lt; heights.length &amp;&amp; y &gt;= 0 &amp;&amp; y &lt; heights[0].length &amp;&amp; 
           heights[r][c] &lt;= heights[x][y]){
            dfs(heights, can_reach, x, y);
        }
    }
}
</code></pre>
<h4 id="permutations">46. Permutations</h4>
<pre><code>public List&lt;List&lt;Integer&gt;&gt; permute(int[] nums) {
    List&lt;List&lt;Integer&gt;&gt; ans = new LinkedList&lt;&gt;();
    
    ArrayList&lt;Integer&gt; nums_list = new ArrayList&lt;Integer&gt;();
    for (int num : nums) nums_list.add(num);
    
    backtracking(nums_list, 0, ans);
    return ans;
}

private void backtracking(ArrayList&lt;Integer&gt; nums, int level, List&lt;List&lt;Integer&gt;&gt; ans){
    if(level == nums.size() - 1){
        ans.add(new ArrayList&lt;Integer&gt;(nums));
        return;
    }
    
    for(int i = level; i &lt; nums.size(); i++){
        Collections.swap(nums, i, level);
        backtracking(nums, level + 1, ans);
        Collections.swap(nums, i, level);
    }
}


public List&lt;List&lt;Integer&gt;&gt; combine(int n, int k) {
    List&lt;List&lt;Integer&gt;&gt; ans = new LinkedList&lt;&gt;();
    
    int[] comb = new int[k];
    int count = 0;
    
    backtracking(ans, comb, count, 1, n, k);
    return ans;
}
</code></pre>
<h4 id="combinations">77. Combinations</h4>
<pre><code>private void backtracking(List&lt;List&lt;Integer&gt;&gt; ans, int[] comb, int count, int pos, int n, int k){
    if(count == k){
        List&lt;Integer&gt; a = new ArrayList&lt;&gt;();
        for(int c : comb) a.add(c);
        ans.add(a);
        return ;
    }
    for(int i = pos; i &lt;= n; i++){
        comb[count++] = i;
        backtracking(ans, comb, count, i + 1, n, k);
        count--;
    }
}
</code></pre>
<p>或者是这样写，Java处理array和list之间的转换比较麻烦</p>
<pre><code>public List&lt;List&lt;Integer&gt;&gt; combine(int n, int k) {
    List&lt;List&lt;Integer&gt;&gt; ans = new ArrayList();
    backTracking(ans, new ArrayList(), 1, n, k);    
    return ans;
}

 public void backTracking(List&lt;List&lt;Integer&gt;&gt; ans, List&lt;Integer&gt; comb, int pos, int n, int k){
     if(comb.size() == k){
         ans.add(new ArrayList(comb));
     }

     for(int i = pos; i &lt;= n; ++i){
            comb.add(i);
            backTracking(ans, comb, i + 1, n, k);
            comb.remove(comb.size() - 1);
        }
 }
</code></pre>
<p>又或者这样</p>
<pre><code>public List&lt;List&lt;Integer&gt;&gt; combine(int n, int k) {
    List&lt;List&lt;Integer&gt;&gt; ans = new LinkedList&lt;&gt;();
    
    List&lt;Integer&gt; comb = new ArrayList&lt;&gt;();
    int count = 0;
    
    backtracking(ans, comb, count, 1, n, k);
    return ans;
}

private void backtracking(List&lt;List&lt;Integer&gt;&gt; ans, List&lt;Integer&gt; comb, int count, int pos, int n, int k){
    if(count == k){
        ans.add(new ArrayList&lt;Integer&gt;(comb));
        return ;
    }
    for(int i = pos; i &lt;= n; i++){
        comb.add(i);
        count++;
        backtracking(ans, comb, count, i + 1, n, k);
        comb.remove(--count);
    }
}
</code></pre>
<h4 id="word-search">79. Word Search</h4>
<pre><code>public boolean exist(char[][] board, String word) {
    if(board.length == 0) return false;
    int m = board.length, n = board[0].length;
    boolean[][] visited = new boolean[m][n];
    boolean[] find = new boolean[]{false};
    
    for(int i = 0; i &lt; m; i++){
        for(int j = 0; j &lt; n; j++){
            backtracking(i, j, board, word, find, visited, 0);
        }
    }
    return find[0];
}

private void backtracking(int i, int j, char[][] board, String word, boolean[] find, boolean[][] visited, int pos){
    if(i &lt; 0 || i &gt;= board.length || j &lt; 0 || j &gt;= board[0].length){
        return ;
    }
    
    if(visited[i][j] || find[0] == true || board[i][j] != word.charAt(pos)){
        return ;
    }
    
    if(pos == word.length() - 1){
        find[0] = true;
        return ;
    }
    
    visited[i][j] = true;
    //递归子节点
    backtracking(i + 1, j, board, word, find, visited, pos + 1);
    backtracking(i - 1, j, board, word, find, visited, pos + 1);
    backtracking(i, j + 1, board, word, find, visited, pos + 1);
    backtracking(i, j - 1, board, word, find, visited, pos + 1);
    visited[i][j] = false;
}
</code></pre>
<p>或者这样写</p>
<pre><code>public boolean exist(char[][] board, String word) {
    for(int i = 0; i &lt; board.length; i ++){
        for(int j = 0; j &lt; board[0].length; j++){
            if(word.charAt(0) == board[i][j] &amp;&amp; backtracking(i, j, 0, word, board)){
                return true;
            }
        }
    }
    return false;
}

public boolean backtracking(int i, int j, int index, String word, char[][] board){
    if(index == word.length()) return true;
    
    if(i &lt; 0 || i &gt;= board.length || 
       j &lt; 0 || j &gt;= board[i].length || 
       word.charAt(index) != board[i][j] || board[i][j] == '0'){
        return false;
    }
    char temp = board[i][j];
    board[i][j] = '0';
    if(backtracking(i + 1, j, index + 1, word, board) ||
       backtracking(i - 1, j, index + 1, word, board) ||
       backtracking(i, j + 1, index + 1, word, board) ||
       backtracking(i, j - 1, index + 1, word, board)){
        return true;
    }
    board[i][j] = temp;
    return false;
}
</code></pre>
<h4 id="n-queens">51. N-Queens</h4>
<pre><code>public List&lt;List&lt;String&gt;&gt; solveNQueens(int n) {
    List&lt;List&lt;String&gt;&gt; ans = new LinkedList&lt;&gt;();
    if(n == 0){
        return ans;
    }
    
    char[][] board = new char[n][n];
    for(int i = 0; i &lt; n; i++){
        for(int j = 0; j &lt; n; j++){
            board[i][j] = '.';
        }
    }
    
    boolean[] column = new boolean[n];
    boolean[] ldiag = new boolean[2 * n - 1];
    boolean[] rdiag = new boolean[2 * n - 1];
    backtracking(ans, board, column, ldiag, rdiag, 0, n);
    return ans;
}

private void backtracking(List&lt;List&lt;String&gt;&gt; ans,  char[][] board, boolean[] column, boolean[] ldiag, boolean[] rdiag, int row, int n){
    if(row == n){
        List&lt;String&gt; str = new LinkedList&lt;&gt;();
        for(int i = 0; i &lt; n; i++){
            StringBuilder sb = new StringBuilder();
            for(int j = 0; j &lt; n; j++){
                sb.append(board[i][j]);
            }
            str.add(sb.toString());
        }
        ans.add(new LinkedList&lt;String&gt;(str));
        return ;
    }
    
    for(int i = 0; i &lt; n; i++){
        if(column[i] || ldiag[n - row + i - 1] || rdiag[row + i]){
            continue;
        }
        
        board[row][i] = 'Q';
        column[i] = ldiag[n - row + i - 1] = rdiag[row + i] = true;
        backtracking(ans, board, column, ldiag, rdiag, row + 1, n);
        board[row][i] = '.';
        column[i] = ldiag[n - row + i - 1] = rdiag[row + i] = false;
    }
}
</code></pre>

