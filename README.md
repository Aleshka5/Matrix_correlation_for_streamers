# Матрица корелляции между играми, которые смотрел пользователь и которые ему предлагает рекоммендательная система
<h1>Введение</h1>
Недавно у меня бьла задача по созданию рекоммендательной системы для компании <a href="https://goodgame.ru/">GoodGame.ru</a>. Подробнее об этом кейсе <a href="https://neural-university.ru/goodgame">здесь</a>.<br><br>
Основной задачей было содать алгоритм на основе глубокого обучения для рекоммендации стримеров в реальном времени индивидуально для каждого пользователя. Именно в процессе создания своего решения я и обнаружил, что существует не так много метрик для априорной оценки разработанного алгоритма рекоммендаций.<br><br>
Всё это сподвигло меня на переосмысление нашей первостепенной задачи. Есть задачи, в которых достаточно трудно или неоднозначно иметь числовую метрику. Так, пердпочтения одного человека не могут быть лучше или хуже, чем предпочтения другого. Они просто разные. В свою очередь от рекоммендательной системы мы хоим наибольшей точности в распознавании этих патернов интересов различных людей. Этот парадокс (что от сети мы хотим универсальности, а для её обучения нам необходима точность) сподвиг меня разобраться в том, какие внутренние зависимости имеет модель глубокого обучения, чтобы лучше понять её рекоммендации. 
<h2>Алгоритм расчёта метрики:</h2>
Если вы хорошо знакомы с методом построения корреляционных матриц, то следующий блок вы можете пропустить за ненадобностью.<br>
Для создания матрицы корреляции я проводил следующий алгоритм:
<ol>
  <li>Во первых модель глубокого обучения делает предсказание потенциально интересных игр для конкретного пользователя (Получаем список <b>A</b>, который на рисунке обозначен синим цветом).</li>
  <li>Затем мы считываем историю ранее просмотренных игр (Получаем список <b>B</b>, который на рисунке подписан как игры интересные пользователю).</li>
  <li>Затем каждой игре из списка <b>B</b> в соответствие ставится массив, в который помещаются все игры из списка <b>A</b>.</li>
  <li>Этот список из векторов может дополняться при предсказании для другого пользователя, где меняется список <b>A</b> и может дополняться список <b>B</b> (Похожие игры будут ссумироваться).</li>
  <li>В итоге получаем матрицу games x games, в которой строки - игры которые были в историях пользователей ранее, а столбцы - игры, которые выдавала рекоммендательная система.</li>
</ol>
<b>Логический смысл этой матрицы в том, чтобы понять как игры из предыдущего опыта пользователя влекут за собой рекоммендации определённых игр.</b>
<img align="middle" src="https://github.com/Aleshka5/Matrix_correlation_for_streamers/blob/main/imgs/Опимание%20метрики_v1.jpg" alt="Картина с описанием работы матрицы корреляции" width=70%>
<h2>Примеры использования</h2>
<b>Note:</b> Очевидно, что данные изображения не иллюстрируют никаких корееляций с базой предоставляемой GoodGame. На них лишь то, как данные обрабатывает та модель глубокого обучения, которая на тот момент была готова. А так же на ней в силу особенностей визуализации не представленные все игры, а лишь их самая популярная часть.<br>
К слову все эти изображения вы можете посмотреть в оригинальном разрешении в этом репозитории в папке <a href="https://github.com/Aleshka5/Matrix_correlation_for_streamers/tree/main/imgs">imgs</a>
<img align="middle" src="https://github.com/Aleshka5/Matrix_correlation_for_streamers/blob/main/imgs/Пример%20метрики1_v1.jpg" alt="Картинка с примером работы матрицы корреляции" width=100%>
Пояснения: пересечения двух игр - квадратик, чем этот квадратик более синий, тем чаще модель глубокого обучения выдавала определённую игру (по столбцу), когда у пользователя в истории была конкретная игра (по строке). Красной линией на картинке обозначена прямая корреляция (там где игры в столбце и в строке совпадают по названию). Так, для майнкрафта, варкрафта и кино прямая корреляциия достаточно велика (т.е. когда пользователь много смотрел эти категории стримов, рекоммендательная система замечала это  рекомендовала их же).<br><br>
Как видно, у неё очень неравномерный вид, не особо понятно по какому принципу происходит рекоммендация. На мой взгляд, это вопрос вольной интерпретации, так как механизм устройства человеческого мозга и его интересов не до конца изучен. Так, я считаю, что для идеально работающей системы рекоммендаций логично было бы увидеть что-то вроде больших кластеров в виде непересекающихся прямоугольников. Это могло бы означать однозначное понимание определённых кластеров в области интересов. Мы их ещё найдём, но об этом чуть позже...
<h2>Нормализация визуализации</h2>
На предыдущей картинке мы увидели много почи белых квадратиков, что сильно режет глаз для визуализвации даннх. Чтобы это исправить мне было достаточно нормализовать данные по частоте их появления в датасете, ведь логично, что чем более популярная игра - тем чаще мы будем в её вектор добавлять наши предсказания.<br><br>
Так, поделив каждую строку на популярность игры я получил следующий результат:
<img align="middle" src="https://github.com/Aleshka5/Matrix_correlation_for_streamers/blob/main/imgs/Пример%20нормализованной%20метрики1_v1.jpg" alt="Картинка с нормализованным примером работы МК" width=100%>
Как видно, остались пробелы в столбцах, но исправлять их - это уже искривление данных, так как рекоммендательная система так же должна иметь понимание о более популярных и более нишевых играх. Так, можно заметить, что игры по типу WarCraft или The Witcher 3 являются очень популярными для рекоммендателной системы, они присутствуют почти в большинстве рекоммендаций. А, например, Dark Souls 3 имеет большую локальную корелляцию с Minecraft и Cult of the Lamb. Dota 2 же оказалась достаточно редкой для рекоммендаций игрой.<br><br>
Этих визуализаций уже достаточно, чтобы понять приоритеты НС и сделать определённые выводы, но дальше - больше
<h1>Метрика V2</h1>
<h2>Введение</h2>
У прошлой метрики была всё-таки одна проблема, которую так и наравило исправить - сортировка по самым похожим векторам. Ранее я говрил о разбиении пространства матрицы на отдельные прямоугольники и вот мы подошли к этому.<br><br>
Для начала скажу пару слов о том как происходит сортировка строк и столбцов таким образом, чтобы рядом стояли наиболее похожие элементы. Это делается в 4 этапа:
<ol>
  <li>Ищем самую популярную игру по столбцу и ставим её на место самой левой (с индексом i = 0).</li>
  <li>Ищем наиболее похожий столбец и ставим его следующим с индексом j = i+1. Так повторяем для всех.</li>
  <li>Ищем самую популярную игру по строке и ставим её на место самой верхней (с индексом i = 0).</li>
  <li>Ищем наиболее похожую строку и ставим её следующей с индексом j = i+1. Так повторяем для всех.</li>
</ol>
<img align="middle" src="https://github.com/Aleshka5/Matrix_correlation_for_streamers/blob/main/imgs/описание%20метрики_v2.JPG" alt="описание метрики_v2" width=100%>
<h2>Примеры</h2>
Вот пример для выполненных этапов 1 и 2:<br>*Важное упоминание, что тут пропадает главная диагональ и прямую корреляцию по этой визуализации отследить невозможно*
<img align="middle" src="https://github.com/Aleshka5/Matrix_correlation_for_streamers/blob/main/imgs/Пример%20работы%20метрики_сорт_X_3_v2.jpg" alt="Пример работы метрики_сорт_X_3_v2.jpg" width=100%>
Тут мы уже можем заметить выстраивание данных в некоторые группы. Так, ярко выделилась группа с общей темой:"Кино" (На изображении выделено красным цветом). Несколько рекоммендаций имеют буквально одинаковые триггеры.<br><br>
А так же можно заметить что самые популярные игры расположилсь по бокам, а "нишевые" по центру, как в некотором скрытом пространстве.<br><hr>
А вот пример для выполненных всех этапов сортировки строк и столбцов:
<img align="middle" src="https://github.com/Aleshka5/Matrix_correlation_for_streamers/blob/main/imgs/Пример%20работы%20метрики_сорт_XY_4_v2.jpg" alt="Пример работы метрики_сорт_XY_4_v2" width=100%>
Внутри красного прямоугольника получилась пустая зона. Она равна той, что была на прошлом изображении. В силу большого количества игр их сложно уместить на одном экране. Тут мы уже можем видеть похожие игры не только по предсказаниям, но и по интересам. Это можно интерпретировать как то, что пользователи, которые смотрят соседние игры, могут иметь похожие предсказания. Тут ещё лучше видно область более редких игр по середине.

<hr>
На следующих двух рисунках вы можете увидеть разницу между матрицами корреляций для моделей с разными кластерезаторами.

### 20 Кластеров
<img align="middle" src="https://github.com/Aleshka5/Matrix_correlation_for_streamers/blob/main/imgs/Пример%20работы%20метрики5_20кластеров_v2.jpg" alt="Пример работы метрики_20кластеров" width=100%>

### 50 Кластеров
<img align="middle" src="https://github.com/Aleshka5/Matrix_correlation_for_streamers/blob/main/imgs/Пример%20работы%20метрики5_50кластеров_v2.jpg" alt="Пример работы метрики_50кластеров" width=100%>
Как вы можете заметить по выделенным областям, для модели с бо'льшим количеством кластеров присущи более яркие точки. Это означает увеличение акцента на локальных группах игр.<br><hr>
На следующих двух рисунках можно также увидеть, что при большем количестве кластеров локальные группы более уверенные и более полные. 
<img align="middle" src="https://github.com/Aleshka5/Matrix_correlation_for_streamers/blob/main/imgs/Пример%20работы%20метрики6_20кластеров_v2.jpg" alt="Пример работы метрики 20 кластеров сорт X" width=100%>
<img align="middle" src="https://github.com/Aleshka5/Matrix_correlation_for_streamers/blob/main/imgs/Пример%20работы%20метрики6_50кластеров_v2.jpg" alt="Пример работы метрики 50 кластеров сорт X" width=100%>
<hr>
Ну и в завершение приведу рядом два примера с сорторовкой по обеим осям для 20ти и 50ти кластеров.<br>

### l_________________________20______________________________________________50_____________________l
<img align="middle" src="https://github.com/Aleshka5/Matrix_correlation_for_streamers/blob/main/imgs/Пример%20работы%20метрики7_20_50_сорт_XY_v2.jpg" alt="Пример работы метрики 20 и 50 кластеров сорт" width=100%>

Здесь вы можете заметить трудно судить о каких-либо прямых соответствиях, так как из-за сортировок все игры сильно перемешаны, но общие черты можно охарактеризовать так: у модели с меньшим количеством кластеров будет более "инертный" переход из одной группы к другой в силу его обобщающих особенностей.
<h2>Выводы</h2>
Результаты моей работы можно охарактеризовать следующим образом. Матрица корреляций позволяет понять закономерности внути вашей системы рекоммендаций и вывести их в визуальное представление. Так как задача рекоммендаций не тривиальна и до конца не изучена, эта работа может помочь вам лучше разобраться в устройстве вашей рекоммендательной системы.
