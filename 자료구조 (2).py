import tkinter as tk
from tkinter import messagebox

window = tk.Tk()
window.title('ALGORITHM')
window.geometry('600x400-500+200')

label = tk.Label(window, text="ALGORITHM", bg='blue', fg='yellow', width=10, height=3, font=('맑은고딕', 25), padx=60)
label.pack()

frame = tk.Frame(window)
frame.pack(padx=10, pady=10)

variable = tk.StringVar()
variable.set("종류 선택")

label = tk.Label(frame, textvariable=variable)
label.grid(row=0, column=0, columnspan=2)

option_menu = tk.OptionMenu(frame, variable, "해시", "트리", "정렬", "최단 경로", "문자열 매칭", "알고리즘")
option_menu.grid(row=1, column=0)

# 타이핑 효과를 위한 함수
def typing_effect(text_widget, text):
    for char in text:
        text_widget.insert(tk.END, char)
        text_widget.update()
        time.sleep(0.05)  # 타이핑 속도 조절

def show_selected_value():
    selected_value = variable.get()
    if selected_value == "정렬":
        open_algorithm_window("정렬", ["버블 정렬", "선택 정렬", "삽입 정렬", "병합 정렬", "퀵 정렬", "기수 정렬", "계수 정렬", "도수 정렬", "힙 정렬","셸 정렬"])
    elif selected_value == "트리":
        open_algorithm_window("트리", ["이진 트리", "이진 탐색 트리", "AVL 트리", "레드-블랙 트리"])
    elif selected_value == "해시":
        open_algorithm_window("해시", ["해시 테이블"])
    elif selected_value == "최단 경로":
        open_algorithm_window("최단 경로", ["다익스트라", "플로이드 와샬 알고리즘"])
    elif selected_value == "문자열 매칭":
        open_algorithm_window("문자열 매칭", ["브루트-포스법","KMP법","Boyer-Moore법"])
    elif selected_value == "알고리즘":
        open_algorithm_window("알고리즘", ["A*", "상태 공간 알고리즘", "유전자 알고리즘"])

button1 = tk.Button(frame, text="확인", command=show_selected_value, padx=15)
button1.grid(row=1, column=1)

window.bind("<Escape>", lambda event: window.destroy())

def handle_key_event(event):
    if event.keysym == "Escape":
        window.withdraw()
    else:
        messagebox.showinfo(message=event.keysym)
    window.bind("<key>", handle_key_event)

def open_algorithm_window(algorithm_type, algorithms):
    algorithm_window = tk.Toplevel(window)
    algorithm_window.title(f"{algorithm_type} 알고리즘 선택")
    algorithm_window.geometry('600x400-500+200')

    frame = tk.Frame(algorithm_window)
    frame.pack(fill=tk.BOTH, expand=True)

    listbox = tk.Listbox(frame, height=1)  # 선택 목록 창의 행 수를 조정
    for algorithm in algorithms:
        listbox.insert(tk.END, algorithm)
    listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    scrollbar = tk.Scrollbar(frame, command=listbox.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    listbox.config(yscrollcommand=scrollbar.set)

    text_box = tk.Text(frame, height=100, width=200)  # 텍스트 박스 생성
    text_box.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

    def show_algorithm_info():
        algorithm_name = listbox.get(tk.ACTIVE)
        if algorithm_name in algorithms_info:
            algorithm_info = algorithms_info[algorithm_name]
            text_box.delete("1.0", tk.END)  # 기존 텍스트 삭제
            text_box.insert(tk.END, algorithm_info)  # 텍스트 박스에 알고리즘 정보 삽입

            
    listbox.bind("<Double-Button-1>", lambda event: show_algorithm_info())

algorithms_info = {
    "버블 정렬":"<개념 및 정의>\n""버블 정렬은 인접한 원소를 두 댁 비교하여 자리를 교환하는 방식을 정복하면서 정렬하는 알고리즘. 버블 정렬을 수행하면 인접한 처음 두 개 원소부터 인접한 마지막 원소까지 비교하는 작업과 자리를 교환하는 작업을 반복하면서 가장 큰 원소가 마지막 자리에 정렬된다."
    "\n\n\n[알고리즘 코드]\n""""def bubble_sort(arr):
  for i in range(len(arr) - 1):
    for j in range(len(arr) - i - 1):
      if arr[j] > arr[j + 1]:
        arr[j], arr[j + 1] = arr[j + 1], arr[j]
  return arr""",
    "선택 정렬":"<개념 및 정의>\n""선택 정렬은 전체 원소 중에서 기준 위치에 맞는 원소를 선택해 자리를 교환하는 방식을 사용. 전체 원소 중에서 가장 작은 원소를 찾은 다음 첫째 원소와 자리를 교환한다. 그 다음 둘째로 작은 원소를 찾고 둘째 원소와 자리를 교환하고, 그 다음에는 셋째로 작은 원소를 찾고 셋째 원소와 자리를 교환한다. 이 과정을 반복하여 정렬을 완성한다."
    "\n\n\n[알고리즘 코드]\n""""def selection_sort(array):
    for i in range(len(array)):
        min_idx = i
        for j in range(i + 1, len(array)):
            if array[j] < array[min_idx]:
                min_idx = j
        array[i], array[min_idx] = array[min_idx], array[i]

    return array

array = [10, 5, 2, 9, 1, 8]
print(selection_sort(array))""",
    "삽입 정렬":"<개념 및 정의>\n""삽입 정렬(Insert Sort)은 정렬되어 있는 부분 집합에 정렬할 새로운 원소의 위치를 찾아 삽입하는 방식으로 정렬 수행. 삽입 정렬에서는 정렬할 자료가 두 개의 부분집합 S(Sotred Subset), U(Unsorted Subset)로 나뉘었으며 정렬된 앞부분의 원소는 S, 아직 정렬하지 않은 나머지 원소는 U. 정렬하지 않은 부분집합 U의 원소를 앞에서부터 하나씩 꺼내서 이미 정렬한 부분 집합 S의 마지막 원소부터 비교하면서 위치를 찾아 삽입하는 방식. 이 정렬을 수행할 때마다 부분집합 S는 늘어나는 방면 U는 줄어든다. 결국, U의 원소를 모두 삽입하여 U가 공집합이 되면 삽입 정렬 완성."
    "\n\n\n[알고리즘 코드]\n""""
def insertion_sort(array):
    for i in range(1, len(array)):
        j = i - 1
        key = array[i]
        while j >= 0 and array[j] > key:
            array[j + 1] = array[j]
            j -= 1
        array[j + 1] = key

    return array

array = [10, 5, 2, 9, 1, 8]
print(insertion_sort(array))
""", 
    "병합 정렬":"<개념 및 정의>\n""병합 정렬(merge Sort)은 정렬을 마친 배열의 병합을 응용하여 분할 정복법에 따라 정렬하는 알고리즘. 먼저 배열을 앞부분과 뒷부분으로 나누고 나눈 두 배열을 각각 정렬하고 병합하면 배열 모두를 정렬할 수 있다."
    "\n\n\n[알고리즘 코드]\n""""def merge_sort(array):
    if len(array) <= 1:
        return array

    mid = len(array) // 2
    left = merge_sort(array[:mid])
    right = merge_sort(array[mid:])

    return merge(left, right)

def merge(left, right):
    result = []
    i, j = 0, 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    result += left[i:]
    result += right[j:]
    return result

array = [10, 5, 2, 9, 1, 8]
print(merge_sort(array))""",
    "퀵 정렬":"<개념 및 정의>\n""퀵 정렬(Quick Sort)은 정렬할 전체 원소에 대해 정렬을 수행하지 않고 기준값을 중심으로 왼쪽 부분집합과 오른쪽 부분집합으로 분할한다. 왼쪽 부분집합에는 기준값보다 작은 원소를 이동시키기고, 오른쪽 부분집합에는 기준값보다 큰 원소를 이동시킨다. 이때 사용하는 기준값을 피벗(Pivot)이라고 하며, 피벗은 전체 원소중 가운데 위치한 원소 또는 첫째 원소 또는 마지막 원소로 정할 수도 있고, 별도의 수식을 사용하여 정하기도 한다."
    "\n\n\n[알고리즘 코드]\n""""def quick_sort(array):
    if len(array) <= 1:
        return array

    pivot = array[len(array) // 2]
    smaller = [x for x in array if x < pivot]
    larger = [x for x in array if x >= pivot]

    return quick_sort(smaller) + [pivot] + quick_sort(larger)

array = [10, 5, 2, 9, 1, 8]
print(quick_sort(array))
""",
    "기수 정렬":"<개념 및 정의>\n""기수 정렬(Radix Sort)은 비교 연산을 사용하지 않고, 데이터를 정렬하는 알고리즘. 데이터를 자릿수별로 비교하여 정렬하는 방식으로 동작하며, 가장 작은 자릿수부터 시작하여 큰 자릿수까지 반복하여 정렬을 수행한다."
    "\n\n\n[알고리즘 코드]\n""""def radix_sort(nums):
    # 자릿수 구하기
    max_digit = 0
    for num in nums:
        max_digit = max(max_digit, len(str(num)))

    # 각 자릿수 정렬
    for i in range(max_digit):
        # 버킷 초기화
        buckets = [[] for _ in range(10)]

        # 각 요소의 해당 자릿수 구하기
        for num in nums:
            digit = int(num / (10 ** i)) % 10
            buckets[digit].append(num)

        # 버킷에서 요소를 순차적으로 꺼내어 원래 배열에 저장
        for bucket in buckets:
            for num in bucket:
                nums.remove(num)
                nums.append(num)

    return nums""",
    "계수 정렬":"<개념 및 정의>\n""계수 정렬(Counting Sort)은 비교 연산을 사용하지 않고, 정수형 데이터를 정렬하는 알고리즘. 데이터의 값들을 카운트하여 각 값의 등장 횟수를 세고, 이를 이용하여 정렬을 수행한다. 데이터의 범위가 제한되어 있을 때 효과적인 정렬 알고리즘."
    "\n\n\n[알고리즘 코드]\n""""def counting_sort(nums):
    # 최대값 구하기
    max_value = max(nums)

    # 빈도수 배열 초기화
    count = [0] * (max_value + 1)

    # 빈도수 계산
    for num in nums:
        count[num] += 1

    # 빈도수에 따라 요소 삽입
    i = 0
    for j in range(max_value + 1):
        for _ in range(count[j]):
            nums[i] = j
            i += 1

    return nums""", 
    "도수 정렬":"<개념 및 정의>\n""두 요소의 키 값을 비교해야 하는 다른 정렬 알고리즘과 다르게 도수 정렬은 요소를 비교할 필요가 없다는 특징이 있다. 도수 정렬 알고리즘은 도수 분포표, 누적도수분포표 작성, 목적 배열 만들기, 배열 복사의 4단계로 이루어진다."
    "\n\n\n[알고리즘 코드]\n""""def counting_sort(nums):
    # 최대값 구하기
    max_value = max(nums)

    # 빈도수 배열 초기화
    count = [0] * (max_value + 1)

    # 빈도수 계산
    for num in nums:
        count[num] += 1

    # 빈도수에 따라 요소 삽입
    i = 0
    for j in range(max_value + 1):
        for _ in range(count[j]):
            nums[i] = j
            i += 1

    return nums""",
    "힙 정렬":"<개념 및 정의>\n""힙 정렬(heap sort)은 힙을 사용하여 정렬하는 알고리즘. 힙(heap)은 '부모의 값이 자식의 값보다 항상 크다.'는 조건을 만족하는 완전이진트리이다. 이때 부모의 값이 자식보다 항상 작아도 힙이라고 한다(부모와 자식 요소의 관계만 일정하면 됨)."
    "\n\n\n[알고리즘 코드]\n""""def heapify(arr, n, i):
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2

    if left < n and arr[left] > arr[largest]:
        largest = left

    if right < n and arr[right] > arr[largest]:
        largest = right

    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)

def heap_sort(arr):
    n = len(arr)

    # 힙으로 변환
    for i in range(n // 2, -1, -1):
        heapify(arr, n, i)

    # 힙에서 루트 노드를 맨 마지막 요소와 교환하면서 정렬
    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        heapify(arr, i, 0)

    return arr""",
    "셸 정렬":"<개념 및 정의>\n""셸 정렬(Shell Sort)은 단순 삽입 정렬의 장점은 살리고 단점은 보완한 정렬 알고리즘. 먼저 정렬할 배열의 요소를 그룹으로 나눠 각 그룹별로 단순 삽입 정렬을 수행하고, 그 그룹을 합치면서 정렬을 반복하여 요소의 이동 횟수를 줄이는 방법."
    "\n\n\n[알고리즘 코드]\n""""def shell_sort(arr):
    n = len(arr)

    # 갭 계산
    gap = n // 2
    while gap > 0:
        # 데이터를 듬성듬성 나누기
        for i in range(gap, n):
            temp = arr[i]
            j = i
            while j >= gap and arr[j - gap] > temp:
                arr[j] = arr[j - gap]
                j -= gap
            arr[j] = temp

        # 갭 반으로 줄이기
        gap //= 2

    return arr""",
    "이진 트리":"<개념 및 정의>\n""이진 트리(Binary tree)는 노드들이 최대 두 개의 자식을 갖는 트리 구조이다. 각 노드는 부모-자식 관계를 가지며, 왼쪽 자식과 오른쪽 자식으로 구성된다. 데이터 삽입, 삭제, 탐색 등을 효율적으로 수행할 수 있는 자료구조."
    "\n\n\n[알고리즘 코드]\n""""class Node:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None


class BinaryTree:
    def __init__(self):
        self.root = None

    def insert(self, data):
        new_node = Node(data)

        if self.root is None:
            self.root = new_node
        else:
            self._insert(new_node, self.root)

    def _insert(self, new_node, current_node):
        if new_node.data < current_node.data:
            if current_node.left is None:
                current_node.left = new_node
            else:
                self._insert(new_node, current_node.left)
        else:
            if current_node.right is None:
                current_node.right = new_node
            else:
                self._insert(new_node, current_node.right)

    def search(self, data):
        if self.root is None:
            return False
        else:
            return self._search(data, self.root)

    def _search(self, data, current_node):
        if current_node.data == data:
            return True
        elif data < current_node.data:
            if current_node.left is None:
                return False
            else:
                return self._search(data, current_node.left)
        else:
            if current_node.right is None:
                return False
            else:
                return self._search(data, current_node.right)

    def inorder(self):
        if self.root is None:
            return
        else:
            self._inorder(self.root)

    def _inorder(self, current_node):
        if current_node is not None:
            self._inorder(current_node.left)
            print(current_node.data)
            self._inorder(current_node.right)

    def preorder(self):
        if self.root is None:
            return
        else:
            self._preorder(self.root)

    def _preorder(self, current_node):
        if current_node is not None:
            print(current_node.data)
            self._preorder(current_node.left)
            self._preorder(current_node.right)

    def postorder(self):
        if self.root is None:
            return
        else:
            self._postorder(self.root)

    def _postorder(self, current_node):
        if current_node is not None:
            self._postorder(current_node.left)
            self._postorder(current_node.right)
            print(current_node.data)""",
    "이진 탐색 트리":"<개념 및 정의>\n""이진 탐색 트리(Binary Search Tree)는 이진 트리의 일종으로, 효율적인 탐색을 위해 정렬된 상태를 유지하는 트리이다. 각 노드는 왼쪽 서브트리의 모든 값보다 크고, 오른쪽 서브트리의 모든 값보다 작다. 탐색, 삽입, 삭제 등을 평균적으로 O(log n)의 시간 복잡도로 수행할 수 있다."
    "\n\n\n[알고리즘 코드]\n""""def binary_search(array, target):
    left = 0
    right = len(array) - 1

    while left <= right:
        mid = (left + right) // 2

        if array[mid] == target:
            return mid
        elif array[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1

array = [1, 3, 5, 7, 9]
target = 5

print(binary_search(array, target))""",
    "AVL 트리":"<개념 및 정의>\n\n""AVL트리는 이진 탐색 트리의 일종으로, 균형 잡힌 트리 구조를 유지하는 트리. 각 노드의 왼쪽 서브트리의 높이 차이가 최대 1인 특성을 가지며, 삽입, 삭제, 연산 시에 회전 연산을 통해 균형을 유지한다."
    "\n\n\n[알고리즘 코드]\n\n""""class AVLTree:

    def __init__(self):
        self.root = None

    def insert(self, value):
        new_node = AVLNode(value)
        self._insert(new_node)

    def _insert(self, new_node):
        parent = None
        curr = self.root
        while curr is not None:
            parent = curr
            if new_node.value < curr.value:
                curr = curr.left
            else:
                curr = curr.right

        new_node.parent = parent
        if parent is None:
            self.root = new_node
        elif new_node.value < parent.value:
            parent.left = new_node
        else:
            parent.right = new_node

        self._fix_insert(new_node)

    def _fix_insert(self, node):
        while node != self.root and node.parent.balance == -2:
            if node.parent == node.parent.parent.left:
                uncle = node.parent.parent.right
                if uncle.balance == 1:
                    node.parent.balance = 0
                    uncle.balance = 0
                    self._rotate_left(node.parent.parent)
                else:
                    node = node.parent
                    self._rotate_right(node)
            else:
                uncle = node.parent.parent.left
                if uncle.balance == -1:
                    node.parent.balance = 0
                    uncle.balance = 0
                    self._rotate_right(node.parent.parent)
                else:
                    node = node.parent
                    self._rotate_left(node)

        self.root.balance = 0

    def inorder(self):
        if self.root is None:
            return

        stack = []
        curr = self.root
        while True:
            while curr is not None:
                stack.append(curr)
                curr = curr.left

            if len(stack) == 0:
                return

            curr = stack.pop()
            print(curr.value)
            curr = curr.right


if __name__ == "__main__":
    numbers = [1, 2, 3, 4, 5]

    tree = AVLTree()
    for number in numbers:
        tree.insert(number)

    tree.inorder()""",
    "레드-블랙 트리":"<개념 및 정의>\n\n""레드-블랙 트리(Red-Black Tree)는 이진 탐색 트리의 일종으로, 균형 잡힌 트리 구조를 유지하는 트리. 각 노드는 레드 또는 블랙의 색상을 가지며, 삽입, 삭제 연산 시에 색상 변경, 회전 연산을 통해 균형을 유지한다."
    "\n\n\n[알고리즘 코드]\n\n""""class RBTree:

    def __init__(self):
        self.root = None

    def insert(self, value):
        new_node = RBNode(value)
        self._insert(new_node)

    def _insert(self, new_node):
        parent = None
        curr = self.root
        while curr is not None:
            parent = curr
            if new_node.value < curr.value:
                curr = curr.left
            else:
                curr = curr.right

        new_node.parent = parent
        if parent is None:
            self.root = new_node
        elif new_node.value < parent.value:
            parent.left = new_node
        else:
            parent.right = new_node

        self._fix_insert(new_node)

    def _fix_insert(self, node):
        while node != self.root and node.parent.color == 'red':
            if node.parent == node.parent.parent.left:
                uncle = node.parent.parent.right
                if uncle.color == 'red':
                    node.parent.color = 'black'
                    uncle.color = 'black'
                    node.parent.parent.color = 'red'
                    node = node.parent.parent
                else:
                    if node == node.parent.right:
                        node = node.parent
                        self._left_rotate(node)
                    node.parent.color = 'black'
                    node.parent.parent.color = 'red'
                    self._right_rotate(node.parent.parent)
            else:
                uncle = node.parent.parent.left
                if uncle.color == 'red':
                    node.parent.color = 'black'
                    uncle.color = 'black'
                    node.parent.parent.color = 'red'
                    node = node.parent.parent
                else:
                    if node == node.parent.left:
                        node = node.parent
                        self._right_rotate(node)
                    node.parent.color = 'black'
                    node.parent.parent.color = 'red'
                    self._left_rotate(node.parent.parent)
        self.root.color = 'black'

    def inorder(self):
        if self.root is None:
            return

        stack = []
        curr = self.root
        while True:
            while curr is not None:
                stack.append(curr)
                curr = curr.left

            if len(stack) == 0:
                return

            curr = stack.pop()
            print(curr.value)
            curr = curr.right


if __name__ == "__main__":
    numbers = [1, 2, 3, 4, 5]

    tree = RBTree()
    for number in numbers:
        tree.insert(number)

    tree.inorder()""",
    "해시 테이블":"<개념 및 정의>\n\n""해시 테이블(Hash Table)은 해시 테이블은 키-값 쌍을 저장하는 자료구조. 각 키를 해시 함수를 통해 고유한 인덱스로 변환하여 값을 저장하고 검색한다. 평균적으로 상수 시간(O(1))에 삽입, 삭제, 검색이 가능한 빠른 자료구조."
    "\n\n\n[알고리즘 코드]\n\n""""class HashTable:

    def __init__(self, size):
        self.size = size
        self.table = [None] * size

    def hash(self, key):
        return key % self.size

    def insert(self, key, value):
        index = self.hash(key)
        node = self.table[index]
        while node is not None:
            if node.key == key:
                node.value = value
                return
            index = (index + 1) % self.size
            node = self.table[index]

        self.table[index] = Node(key, value)

    def get(self, key):
        index = self.hash(key)
        node = self.table[index]
        while node is not None:
            if node.key == key:
                return node.value
            index = (index + 1) % self.size
            node = self.table[index]

        return None


class Node:

    def __init__(self, key, value):
        self.key = key
        self.value = value


if __name__ == "__main__":
    numbers = [1, 2, 3, 4, 5]

    table = HashTable(10)
    for number in numbers:
        table.insert(number, number)

    print(table.get(1))
    print(table.get(2))
    print(table.get(3))
    print(table.get(4))
    print(table.get(5))""",
    "다익스트라":"<개념 및 정의>\n\n""다익스트라 알고리즘(Dijkstra's Algorithm)은 가중 그래프에서 최단 경로를 찾는 알고리즘. 출발점에서부터 각 정점까지의 최단 거리를 계산하고, 최단 거리를 갱신해가며 탐색한다. 음의 가중치를 갖지 않는 그래프에서 사용 가능한 알고리즘."
    "\n\n\n[알고리즘 코드]\n\n""""def dijkstra(graph, start):
    distances = [float("inf")] * len(graph)
    distances[start] = 0
    visited = [False] * len(graph)

    queue = [start]
    while queue:
        u = queue.pop(0)
        visited[u] = True

        for v, weight in graph[u].items():
            if not visited[v] and distances[v] > distances[u] + weight:
                distances[v] = distances[u] + weight
                queue.append(v)

    return distances

graph = {
    0: {1: 10, 2: 5},
    1: {2: 3, 3: 1},
    2: {3: 2},
    3: {},
}

print(dijkstra(graph, 0))""",
    "플로이드 와샬 알고리즘":"<개념 및 정의>\n\n""플로이드-와샬 알고리즘(Floyd-Warshall Algorithm)은 모든 정점 쌍 간의 최단 경로를 찾은 알고리즘. 동적 계획법을 기반으로 하며, 각 정점을 거쳐가는 경우를 고려하여 최단 거리를 갱신하며, 음의 가중치를 갖는 그래프에서도 사용 가능한 알고리즘."
    "\n\n\n[알고리즘 코드]\n\n""""def floyd_warshall(graph):
    n = len(graph)

    for k in range(n):
        for i in range(n):
            for j in range(n):
                if graph[i][j] > graph[i][k] + graph[k][j]:
                    graph[i][j] = graph[i][k] + graph[k][j]

    return graph

graph = {
    0: {1: 10, 2: 5},
    1: {2: 3, 3: 1},
    2: {3: 2},
    3: {},
}

print(floyd_warshall(graph))""",
    "브루트-포스법":"<개념 및 정의>\n\n""문자열을 검색하는 브루트-포스법(brute force method). 문자열 검색이란 어떤 문자열 안에 다른 문자열이 들어 있는지 조사하고 들어 있다면 그 위치를 찾아내는 것."
    "\n\n\n[알고리즘 코드]\n\n""""def brute_force_match(text, pattern):
    i = 0
    j = 0

    while i < len(text) and j < len(pattern):
        if text[i] == pattern[j]:
            i += 1
            j += 1
        else:
            i = i - j + 1
            j = 0

    if j == len(pattern):
        return i - j
    else:
        return -1

text = "This is a test string."
pattern = "test"

print(brute_force_match(text, pattern))""",
    "KMP법":"<개념 및 정의>\n\n""KMP법은 검사했던 위치 결과를 버리지 않고 이를 효율적으로 활용하는 알고리즘."
    "\n\n\n[알고리즘 코드]\n\n""""def kmp_match(text, pattern):
    pi = [0] * len(pattern)
    j = 0
    for i in range(1, len(pattern)):
        while j > 0 and pattern[i] != pattern[j]:
            j = pi[j - 1]
        if pattern[i] == pattern[j]:
            j += 1
        pi[i] = j

    i = 0
    j = 0
    while i < len(text) and j < len(pattern):
        if text[i] == pattern[j]:
            i += 1
            j += 1
        else:
            if j == 0:
                i += 1
            else:
                j = pi[j - 1]

    if j == len(pattern):
        return i - j
    else:
        return -1

text = "This is a test string."
pattern = "test"

print(kmp_match(text, pattern))""",
    "Boyer-Moore법":"<개념 및 정의>\n\n""Boyer-Moore은 텍스트의 내용을 아예 보지 않고 2칸 이상 점프할 수 있도록 하는 알고리즘. 까다로운 부분을 조금 간단하게 처리해도 전체 성능에는 거의 영향을 주지 않는다."
    "\n\n\n[알고리즘 코드]\n\n"""" m = len(pattern)
    n = len(text)

    # 스킵 테이블 생성
    skip_table = [m] * 256
    for i in range(m - 1, -1, -1):
        skip_table[ord(pattern[i])] = m - i - 1

    # 검색 시작
    i = m - 1
    while i < n:
        j = m - 1
        while j >= 0 and text[i - j] == pattern[j]:
            j -= 1

        if j < 0:
            # 패턴을 찾음
            return i - m + 1
        else:
            # 패턴을 찾지 못함
            i += skip_table[ord(text[i])]

    return -1""",
    "A*":"<개념 및 정의>\n\n""A*알고리즘은 그래프 탐색 및 경로 탐색에 사용되는 알고리즘. 휴리스틱 함수를 이용하여 최적의 경로를 찾는 것이 특징이다. 최단 경로 문제 등에 활용되며, 효율적인 경로 탐색을 제공."
    "\n\n\n[알고리즘 코드]\n\n""""from heapq import heappop, heappush
    class Node:
    def __init__(self, state, parent, g_score, h_score):
        self.state = state
        self.parent = parent
        self.g_score = g_score
        self.h_score = h_score
        self.f_score = self.g_score + self.h_score

    def __lt__(self, other):
        return self.f_score < other.f_score

def a_star(start, goal, heuristic, neighbors):
    open_set = [Node(start, None, 0, heuristic(start, goal))]
    closed_set = set()

    while open_set:
        current = heappop(open_set)

        if current.state == goal:
            path = []
            while current:
                path.append(current.state)
                current = current.parent
            return path[::-1]

        closed_set.add(current.state)

        for neighbor, distance in neighbors(current.state):
            if neighbor in closed_set:
                continue

            tentative_g_score = current.g_score + distance
            existing_node = next((node for node in open_set if node.state == neighbor), None)

            if not existing_node or tentative_g_score < existing_node.g_score:
                new_node = Node(neighbor, current, tentative_g_score, heuristic(neighbor, goal))
                if existing_node:
                    open_set.remove(existing_node)
                open_set.append(new_node)

    return None

# Example usage
def heuristic(start, goal):
    # Your heuristic function here
    pass

def neighbors(state):
    # Your neighbor function here
    pass

start = ...  # Start state
goal = ...  # Goal state
path = a_star(start, goal, heuristic, neighbors)

if path:
    print(f"Path found: {path}")
else:
    print("Path not found")""",
    "상태 공간 알고리즘":"<개념 및 정의>\n\n""상태 공간 알고리즘은 상태와 연산자를 이용하여 문제를 해결하는 알고리즘. 상태 공간을 탐색하면서 목표 상태에 도달하는 경로를 찾는 것이 목적이다. 너비 우선 탐색, 깊이 우선 탐색 등이 상태 공간 알고리즘의 일종."
    "\n\n\n[알고리즘 코드]\n\n""""def max_sum_subsequence(numbers):
    ""숫자 배열의 합이 최대인 부분 배열을 찾습니다.

    Args:
        numbers: 숫자 배열

    Returns:
        합이 최대인 부분 배열""

    def dfs(numbers, index, sum):
        if index == len(numbers):
            return sum

        # 현재 상태에서 가능한 모든 상태로 확장합니다.
        next_states = []
        for i in range(index + 1, len(numbers)):
            next_sum = sum + numbers[i]
            next_states.append((i, next_sum))

        # 확장된 상태 중에서 다음으로 탐색할 상태를 선택합니다.
        next_index, next_sum = max(next_states, key=lambda t: t[1])

        return dfs(numbers, next_index, next_sum)

    return dfs(numbers, 0, 0)


if __name__ == "__main__":
    numbers = [1, 2, 3, 4, 5]
    print(max_sum_subsequence(numbers))""",
    "유전자 알고리즘":"<개념 및 정의>\n\n""유전자 알고리즘(Genetic Algorithm)은 진화 이론에서 영감을 받아 최적화 문제를 해결하는 알고리즘. 개체의 유전자를 표현하고, 선택, 교차, 변이 등의 연산을 통해 해를 찾아낸다."
    "\n\n\n[알고리즘 코드]\n\n""""def fitness(numbers):
    return sum(numbers)

def generate_chromosome():
    return [random.choice(numbers) for _ in range(len(numbers))]

def crossover(chromosome1, chromosome2):
    point = random.randint(0, len(chromosome1) - 1)
    return chromosome1[:point] + chromosome2[point:]

def mutate(chromosome):
    index = random.randint(0, len(chromosome) - 1)
    chromosome[index] = random.choice(numbers)

def main():
    numbers = [1, 2, 3, 4, 5]

    population = []
    for _ in range(100):
        population.append(generate_chromosome())

    for _ in range(100):
        # 선택
        parents = sorted(population, key=fitness, reverse=True)[:20]

        # 교배
        child1 = crossover(parents[0], parents[1])
        child2 = crossover(parents[1], parents[0])

        # 변이
        mutate(child1)
        mutate(child2)

        population.append(child1)
        population.append(child2)

    # 최적해를 출력
    best_chromosome = sorted(population, key=fitness, reverse=True)[0]
    print(best_chromosome)

if __name__ == "__main__":
    main()
"""
}

window.mainloop()