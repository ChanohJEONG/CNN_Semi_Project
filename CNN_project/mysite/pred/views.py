from django.shortcuts import render
from django.views.generic import TemplateView
from .forms import ImageForm
from .main import detect

class PredView(TemplateView):
    # 생성자
    def __init__(self):
        self.params = {'result_list':[],
        'result_name':"",
        'result_img':"",
        'form': ImageForm()}
    #GET requests (index.html 파일 초기 표시)
    def get(self, req):
        return render(req, 'pred/index.html', self.params)
        
    #POST requests (index.html 파일에 결과 표시)
    def post(self, req):
        # POST method에 의해 전달되는 Form Data
        form = ImageForm(req.POST, req.FILES)
        # Form Data error check
        if not form.is_valid():
            raise ValueForm('invalid form')
        # Form Data에서 이미지 파일 얻기
        image = form.cleaned_data['image']
        # Image file을 지정해서 얼굴 인식
        result = detect(image)

        # 분류된 얼굴 결과 저장
        self.params['result_list'], self.params['result_name'],self.params['result_img'] = result

        # 페이지에 화면 표시
        return render(req, 'pred/index.html', self.params)